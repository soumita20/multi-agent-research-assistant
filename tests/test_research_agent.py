from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import research_agent
from research_agent import (
    AppConfig,
    Perspectives,
    ResearchAssistant,
    SearchQuery,
    TokenTracker,
    build_assistant,
)


def bare_assistant(llm=None):
    assistant = ResearchAssistant.__new__(ResearchAssistant)
    assistant.llm = llm
    assistant.token_tracker = TokenTracker()
    return assistant


def test_token_tracker_reads_langchain_usage_metadata():
    tracker = TokenTracker()
    tracker.record(
        "sample",
        AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        ),
    )

    assert tracker.totals.total_tokens == 17
    assert tracker.calls[0]["call"] == "sample"


def test_llm_call_retries_429_then_returns_response(monkeypatch):
    class RateLimit(Exception):
        status_code = 429

    calls = iter((RateLimit("Please try again in 0.01s"), "ok"))
    sleeps = []
    monkeypatch.setattr(research_agent.time, "sleep", sleeps.append)

    def invoke():
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    result = ResearchAssistant._invoke_with_rate_limit_retry(
        invoke, "test_call"
    )

    assert result == "ok"
    assert len(sleeps) == 1
    assert sleeps[0] >= 0.5


def test_llm_call_does_not_retry_non_429_errors(monkeypatch):
    class ServerError(Exception):
        status_code = 500

    sleeps = []
    monkeypatch.setattr(research_agent.time, "sleep", sleeps.append)

    with pytest.raises(ServerError):
        ResearchAssistant._invoke_with_rate_limit_retry(
            lambda: (_ for _ in ()).throw(ServerError("failed")), "test_call"
        )

    assert sleeps == []


def test_llm_call_respects_retry_after_header(monkeypatch):
    class RateLimit(Exception):
        status_code = 429
        response = SimpleNamespace(status_code=429, headers={"retry-after": "2"})

    sleeps = []
    monkeypatch.setattr(research_agent.time, "sleep", sleeps.append)
    attempts = {"count": 0}

    def invoke():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimit("limited")
        return "ok"

    assert ResearchAssistant._invoke_with_rate_limit_retry(invoke, "test_call") == "ok"
    assert sleeps == [2.5]


def test_llm_call_does_not_retry_daily_quota_errors(monkeypatch):
    class DailyQuota(Exception):
        status_code = 429

    sleeps = []
    monkeypatch.setattr(research_agent.time, "sleep", sleeps.append)

    with pytest.raises(DailyQuota, match="requests per day limit exceeded"):
        ResearchAssistant._invoke_with_rate_limit_retry(
            lambda: (_ for _ in ()).throw(
                DailyQuota("Generate content requests per day limit exceeded")
            ),
            "test_call",
        )

    assert sleeps == []


def test_structured_call_returns_validated_schema():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content=""),
        "parsed": SearchQuery(search_query="grid investment"),
        "parsing_error": None,
    }
    assistant = bare_assistant(llm)

    result = assistant._invoke_structured(
        SearchQuery, [HumanMessage(content="search")], "query"
    )

    assert result == SearchQuery(search_query="grid investment")


def test_structured_call_returns_none_when_parser_fails():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "raw": AIMessage(content="not structured"),
        "parsed": None,
        "parsing_error": ValueError("bad output"),
    }
    assistant = bare_assistant(llm)

    assert assistant._invoke_structured(Perspectives, [], "analysts") is None


def test_google_model_requests_end_with_user_message_after_model_turn():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="An answer")
    assistant = bare_assistant(llm)
    assistant.llm_provider = "google"

    assistant._invoke(
        [HumanMessage(content="Opening"), AIMessage(content="Analyst question?")],
        "generate_answer",
    )

    sent_messages = llm.invoke.call_args.args[0]
    assert isinstance(sent_messages[-1], HumanMessage)
    assert "Answer the analyst's latest question" in sent_messages[-1].content


def test_non_google_model_messages_are_left_unchanged():
    assistant = bare_assistant(MagicMock())
    assistant.llm_provider = "groq"
    prior_messages = [HumanMessage(content="Opening"), AIMessage(content="Question?")]

    assert assistant._prepare_messages(prior_messages, "generate_answer") is prior_messages


def test_search_query_call_passes_messages_not_prompt_template():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": SearchQuery(search_query="grid investment"),
        "parsing_error": None,
    }
    assistant = bare_assistant(llm)

    query = assistant._resolve_search_query(
        {"messages": [HumanMessage(content="How are grid investments changing?")]},
        "search_query:web",
    )

    sent_messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert isinstance(sent_messages, list)
    assert isinstance(sent_messages[0], SystemMessage)
    assert query == "grid investment"


def test_tavily_auth_failure_is_recorded_as_safe_run_warning():
    assistant = bare_assistant(MagicMock())
    assistant.run_warnings = []

    results = assistant._extract_results(
        {"error": "Error 401 Unauthorized: invalid API key: secret-value"},
        "web search",
    )

    assert results == []
    assert any("authentication failed (HTTP 401)" in warning for warning in assistant.run_warnings)
    assert "secret-value" not in str(assistant.run_warnings)


def test_build_assistant_selects_groq_provider(monkeypatch):
    groq_model = MagicMock()
    monkeypatch.setattr(research_agent, "ChatGroq", MagicMock(return_value=groq_model))
    monkeypatch.setattr(research_agent, "TavilySearch", MagicMock(return_value=object()))

    assistant = build_assistant(
        AppConfig(
            topic="Grid investment",
            llm_provider="groq",
            model_name="openai/gpt-oss-20b",
            api_key="test-key",
        )
    )

    research_agent.ChatGroq.assert_called_once()
    assert assistant.llm is groq_model


def test_build_assistant_keeps_ollama_as_default_provider(monkeypatch):
    ollama_model = MagicMock()
    monkeypatch.setattr(research_agent, "ChatOllama", MagicMock(return_value=ollama_model))
    monkeypatch.setattr(research_agent, "TavilySearch", MagicMock(return_value=object()))

    assistant = build_assistant(
        AppConfig(topic="Grid investment", model_name="qwen2.5:14b")
    )

    research_agent.ChatOllama.assert_called_once()
    assert assistant.llm is ollama_model


def test_build_assistant_requires_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_assistant(AppConfig(topic="Grid investment", llm_provider="groq"))
