"""
research_agent.py
==================

Multi-agent research assistant built on LangGraph.

Pipeline
--------
1. Given a topic, generate a panel of AI "analyst" personas, each covering a
   distinct sub-topic (with an optional human-in-the-loop feedback round).
2. Each analyst conducts a multi-turn interview with an "expert" model. The
   expert answers using context pulled from web search (Tavily) and
   Wikipedia, gathered in parallel for every turn.
3. Each interview is condensed into a report section.
4. All sections are merged (map-reduce, via LangGraph's ``Send`` API) into a
   final report with an introduction, body, and conclusion.

Usage
-----
    python research_agent.py --topic "How can subclinical hypothyroidism be
    controlled?" --max-analysts 3 --output report.md

    # Verbose logging + a custom log file path (default: research_agent.log)
    python research_agent.py --topic "..." --verbose --log-file run.log

    # Disable file logging entirely (console only)
    python research_agent.py --topic "..." --log-file ""

Environment variables required (prompted for interactively if missing):
    LANGSMITH_API_KEY   - optional, enables LangSmith tracing
    TAVILY_API_KEY       - required, powers the web-search tool
"""

from __future__ import annotations

import argparse
import getpass
import logging
import operator
import os
from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_community.document_loaders import WikipediaLoader
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger("research_agent")


#Configuration

@dataclass
class AppConfig:
    """Runtime configuration for a research run."""

    topic: str
    max_analysts: int = 3
    max_interview_turns: int = 2
    model_name: str = "qwen2.5:14b"
    temperature: float = 0.0
    tavily_max_results: int = 3
    thread_id: str = "1"
    output_path: str = "research_report.md"
    interactive_feedback: bool = True
    log_file: Optional[str] = None
    verbose: bool = False
    social_domains: List[str] = field(default_factory=lambda: ["linkedin.com"])
    enable_social_search: bool = True


def configure_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Set up logging for the whole application: console always, plus an
    optional rotating-free file handler when ``log_file`` is given."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicate handlers on repeated calls (e.g. in notebooks)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file)


def _ensure_env_var(var_name: str) -> None:
    """Prompt for and set an environment variable if it isn't already set."""
    if not os.environ.get(var_name):
        os.environ[var_name] = getpass.getpass(f"{var_name}: ")


def load_environment() -> None:
    """Load .env and make sure required API keys are present."""
    load_dotenv()
    _ensure_env_var("LANGSMITH_API_KEY")
    _ensure_env_var("TAVILY_API_KEY")

    for var_name in ("LANGSMITH_API_KEY", "TAVILY_API_KEY"):
        value = os.environ.get(var_name)
        if value:
            logger.info("%s loaded (starts with %s...)", var_name, value[:4])
        else:
            logger.warning("%s was not found; related features may not work.", var_name)


def log_gpu_availability() -> None:
    """Best-effort diagnostic log of CUDA/GPU availability. Never fatal."""
    try:
        import torch  # local import: optional, heavy dependency

        if torch.cuda.is_available():
            logger.info(
                "CUDA available - using GPU: %s (CUDA %s)",
                torch.cuda.get_device_name(0),
                torch.version.cuda,
            )
        else:
            logger.info("CUDA not available; running on CPU.")
    except ImportError:
        logger.debug("torch not installed; skipping GPU diagnostic.")


#Models

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst.")
    description: str = Field(description="Description of the analyst's focus area.")

    @property
    def persona(self) -> str:
        return f"{self.name} is a {self.role} at {self.affiliation}. {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="List of analysts providing perspectives with their roles and affiliations."
    )


class SearchQuery(BaseModel):
    search_query: Optional[str] = Field(
        default=None,
        description="A well-structured search query for retrieval. Must be a plain string.",
    )


#Token tracking

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TokenTracker:
    """
    Accumulates per-call and running-total LLM token usage for a run.
    Reads usage off whatever the provider actually reports: LangChain's
    standard ``AIMessage.usage_metadata`` where available, falling back to
    Ollama's ``prompt_eval_count`` / ``eval_count`` in ``response_metadata``
    for providers that don't populate the standard field.
    """

    def __init__(self) -> None:
        self.calls: List[dict] = []
        self.totals = TokenUsage()

    @staticmethod
    def _extract_usage(response: Any) -> Optional[TokenUsage]:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return TokenUsage(
                input_tokens=usage.get("input_tokens", 0) or 0,
                output_tokens=usage.get("output_tokens", 0) or 0,
                total_tokens=usage.get("total_tokens", 0) or 0,
            )

        meta = getattr(response, "response_metadata", None) or {}
        prompt_tokens = meta.get("prompt_eval_count")
        completion_tokens = meta.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            prompt_tokens = prompt_tokens or 0
            completion_tokens = completion_tokens or 0
            return TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        return None

    def record(self, call_label: str, response: Any) -> None:
        """Log and accumulate token usage for one LLM call. No-op (with a
        debug note) if the provider didn't report usage for this call."""
        usage = self._extract_usage(response)
        if usage is None:
            logger.debug("No token usage metadata available for call: %s", call_label)
            return

        self.calls.append(
            {
                "call": call_label,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        )
        self.totals.input_tokens += usage.input_tokens
        self.totals.output_tokens += usage.output_tokens
        self.totals.total_tokens += usage.total_tokens

        logger.info(
            "Token usage | %-20s | input=%-6d output=%-6d total=%-6d",
            call_label, usage.input_tokens, usage.output_tokens, usage.total_tokens,
        )

    def log_summary(self) -> None:
        logger.info(
            "Token usage TOTAL | calls=%d | input=%d | output=%d | total=%d",
            len(self.calls),
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.totals.total_tokens,
        )


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    interview_topic: str


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: Optional[str]
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


#Prompt templates

ANALYST_INSTRUCTIONS = """You are tasked with creating a diverse set of AI Analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the personas. This feedback may include specific areas of focus, desired expertise, or any other relevant information.
{human_analyst_feedback}

3. Determine the most interesting themes based on the documents and/or feedback provided above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn more about a specific topic.

Your goal is boiling down to interesting and specific insights related to the topic.

1. Interesting: Insights that people will find surprising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your time and insights. I really appreciate it."

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

SEARCH_INSTRUCTIONS = SystemMessage(
    content="""You will be given a conversation between an analyst and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well structured web search query."""
)

ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.
Here is the analyst area of focus: {goals}.
Your goal is to answer a question posed by the interviewer.
To answer a question, use this context: {context}

When answering questions, follow these guidelines:
1. Use only information provided in the context.
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
3. The context contains sources at the top of each individual document.
4. Include these sources in your answer next to any relevant statements. For example, for source # 1 use [1].
5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc.
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/> then just list:
      [1] assistant/docs/llama3_1.pdf, page 7
And skip the addition of the brackets as well as the Document source preamble in your citation."""

SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer.
Your task is to create a short, easily digestible section of a report based on a set of source documents.
1. Analyze the content of the source documents:
- The name of each source document is at the start of the document, with the <Document tag.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst:
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents

6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. Do not repeat the same source under two different numbers.

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They wrote up their findings into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-heading.
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:

{context}
"""

INTRO_CONCLUSION_INSTRUCTIONS = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

Your job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

INTERVIEW_CLOSING_LINE = "Thank you so much for your time and insights. I really appreciate it."


#Construction of the graph

class ResearchAssistant:
    """
    Encapsulates the LangGraph pipeline for the multi-agent research
    assistant. The LLM and search tool are injected (rather than read from
    module-level globals) so the class is easy to test and reconfigure.
    """

    def __init__(
        self,
        llm: Any,
        search_tool: Any,
        max_interview_turns: int = 2,
        checkpointer: Optional[MemorySaver] = None,
        social_domains: Optional[List[str]] = None,
        enable_social_search: bool = True
    ) -> None:
        self.llm = llm
        self.search_tool = search_tool
        self.max_interview_turns = max_interview_turns
        self.checkpointer = checkpointer or MemorySaver()
        self.token_tracker = TokenTracker()
        self.social_domains = list(social_domains or []) if enable_social_search else []

        self._interview_builder = self._build_interview_subgraph()
        self.graph = self._build_research_graph()

    # LLM Call helpers

    def _invoke(self, messages: list, call_label: str):
        """Plain (non-structured) LLM call, with token usage recorded."""
        response = self.llm.invoke(messages)
        self.token_tracker.record(call_label, response)
        return response

    def _invoke_structured(self, schema: type, messages: list, call_label: str):
        """
        Structured-output LLM call. ``with_structured_output`` normally
        returns only the parsed Pydantic object and discards the raw
        AIMessage (and with it, token usage metadata) - requesting the raw
        response alongside the parsed one lets us track usage for these
        calls too. Returns the parsed object (or None on a parsing error,
        same as the caller would see without this wrapper).
        """
        structured_llm = self.llm.with_structured_output(schema, include_raw=True)
        result = structured_llm.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if raw is not None:
            self.token_tracker.record(call_label, raw)
        return parsed

    def log_token_summary(self) -> None:
        """Log total input/output/total token counts for the whole run."""
        self.token_tracker.log_summary()

    # Generate analysts and ask human feedback

    def _generate_analysts(self, state: ResearchGraphState) -> dict:
        """Generate a panel of analyst personas for the given topic."""
        topic = state["topic"]
        max_analysts = state["max_analysts"]
        human_analyst_feedback = state.get("human_analyst_feedback", "")

        system_message = ANALYST_INSTRUCTIONS.format(
            topic=topic,
            human_analyst_feedback=human_analyst_feedback,
            max_analysts=max_analysts,
        )
        result = self._invoke_structured(
            Perspectives,
            [SystemMessage(content=system_message)]
            + [HumanMessage(content="Generate a diverse set of AI Analyst personas based on the above instructions.")],
            call_label="generate_analysts",
        )
        if result is None:
            raise ValueError("Analyst generation returned no parsed result (structured-output parsing error).")

        logger.info("Generated %d analyst persona(s).", len(result.analysts))
        return {"analysts": result.analysts}

    @staticmethod
    def _human_feedback(state: ResearchGraphState) -> None:
        """No-op node; execution pauses here (interrupt_before) for human input."""
        return None

    @staticmethod
    def _route_after_feedback(state: ResearchGraphState):
        """Regenerate analysts on feedback, otherwise fan out interviews."""
        human_analyst_feedback = state.get("human_analyst_feedback")
        if human_analyst_feedback:
            return "generate_analysts"

        topic = state["topic"]
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "interview_topic": topic,
                    "messages": [
                        HumanMessage(content=f"So you said you were writing an article on {topic}?")
                    ],
                },
            )
            for analyst in state["analysts"]
        ]

    # Interview sub graphs

    def _generate_question(self, state: InterviewState) -> dict:
        """The analyst asks the next interview question."""
        analyst = state["analyst"]
        messages = state["messages"]
        system_message = QUESTION_INSTRUCTIONS.format(goals=analyst.persona)
        question = self._invoke(
            [SystemMessage(content=system_message)] + messages, call_label="generate_question"
        )
        return {"messages": [question]}

    def _resolve_search_query(self, state: InterviewState, call_label: str) -> str:
        """
        Ask the LLM for a structured search query. If that fails or returns
        empty/null, fall back to the analyst's focus-area description (free
        - already in state, no extra LLM call) rather than the raw
        conversational message text, which makes for a poor, token-heavy
        search query. As a last resort (no analyst in state), truncate the
        last message to a short excerpt instead of sending it whole.
        """
        try:
            result = self._invoke_structured(
                SearchQuery, [SEARCH_INSTRUCTIONS] + state["messages"], call_label=call_label
            )
            raw_query = getattr(result, "search_query", None)
            query = raw_query.strip() if raw_query else ""
            if not query:
                raise ValueError("Empty or null search_query returned")
            return query
        except (OutputParserException, ValueError, Exception) as exc:
            logger.warning("Search query generation failed (%s); using fallback.", exc)
            return self._fallback_search_query(state)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        """Trim to a short, search-friendly phrase: cut at the first
        sentence boundary if one falls within the word budget, otherwise
        hard-truncate at max_words. Keeps queries free of filler like
        '... is a specialist with over 15 years of experience in ...'."""
        text = text.strip()
        if not text:
            return ""
        first_sentence = text.split(".")[0].strip()
        if first_sentence and len(first_sentence.split()) <= max_words:
            return first_sentence
        return " ".join(text.split()[:max_words])

    def _fallback_search_query(self, state: InterviewState, max_words: int = 12) -> str:
        """
        Cheap, no-LLM-call fallback query, in priority order:
        1. The overall research topic (short, on-topic, set once per run) -
           by far the best fallback, since it's already a clean subject
           phrase rather than prose.
        2. The analyst's focus-area description, truncated to a short
           phrase - a bio paragraph is not itself a search query.
        3. A short excerpt of the last message, as a last resort.
        """
        topic = state.get("interview_topic")
        if topic:
            return self._truncate(topic, max_words)

        analyst = state.get("analyst")
        if analyst is not None and getattr(analyst, "description", None):
            return self._truncate(analyst.description, max_words)

        last_content = state["messages"][-1].content if state.get("messages") else ""
        if not isinstance(last_content, str):
            return ""
        return self._truncate(last_content, max_words)

    def _search_web(self, state: InterviewState) -> dict:
        """Fetch web-search context for the current interview turn via Tavily."""
        if not state.get("messages"):
            return {"context": ["No conversation history available to search."]}

        query = self._resolve_search_query(state, call_label="search_query:web")
        if query.lower() in ("", "none", "null"):
            return {"context": ["Search skipped: no relevant search query needed."]}

        try:
            data = self.search_tool.invoke({"query": query})
            search_docs = data.get("results", data) if isinstance(data, dict) else data
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return {"context": [f"Web search failed for query '{query}'."]}

        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        )
        return {"context": [formatted_docs]}

    @staticmethod
    def _diagnose_wikipedia_failure(query: str, exc: Exception) -> None:
        """
        Best-effort root-cause probe for a WikipediaLoader failure. Calls the
        underlying `wikipedia` package directly (bypassing LangChain's
        wrapper) to figure out whether the empty response is coming from
        Wikipedia's search API itself, or from something downstream (e.g. a
        malformed/too-long query, rate limiting, or a network/proxy issue).
        Never raises - this is diagnostics only.
        """
        logger.debug(
            "Wikipedia fetch failure detail | exc_type=%s | query_len=%d | query=%r",
            type(exc).__name__,
            len(query),
            query[:200],
        )
        try:
            import wikipedia  # the lower-level package langchain wraps

            raw_results = wikipedia.search(query, results=3)
            logger.debug("Raw wikipedia.search() returned: %r", raw_results)
            if not raw_results:
                logger.info(
                    "Wikipedia's search API returned zero results for this query "
                    "(likely too specific/long, or phrased as a question rather "
                    "than a topic) - this is a query-relevance issue, not a bug."
                )
            else:
                # Search worked; the failure is probably in fetching a specific
                # page (disambiguation, redirect loop, or a transient empty
                # response on the page-content call).
                page_title = raw_results[0]
                try:
                    wikipedia.page(page_title, auto_suggest=False)
                    logger.info(
                        "wikipedia.page(%r) succeeded on retry - original failure "
                        "looks transient (rate limit or a dropped response).",
                        page_title,
                    )
                except Exception as page_exc:
                    logger.info(
                        "wikipedia.page(%r) also failed: %s (%s) - likely a "
                        "disambiguation page or a page-fetch issue for this title.",
                        page_title,
                        page_exc,
                        type(page_exc).__name__,
                    )
        except ImportError:
            logger.debug("`wikipedia` package not importable; skipping deeper probe.")
        except Exception as probe_exc:
            logger.debug("Diagnostic probe itself failed: %s", probe_exc)

    def _search_wikipedia(self, state: InterviewState) -> dict:
        """Fetch Wikipedia context for the current interview turn."""
        query = self._resolve_search_query(state, call_label="search_query:wiki")
        logger.debug("Wikipedia query resolved to: %r", query)

        try:
            search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        except Exception as exc:
            logger.warning("Wikipedia fetch failed: %s", exc)
            self._diagnose_wikipedia_failure(query, exc)
            search_docs = []

        if not search_docs:
            logger.info("No Wikipedia context returned for query: %r", query[:200])

        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc.metadata.get("source", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        )
        return {"context": [formatted_docs]}

    def _search_social(self, state: InterviewState) -> dict:
        """
        Fetch context from public, search-engine-indexed pages on employee
        or company social platforms (e.g. LinkedIn), scoped via Tavily's
        include_domains. This queries a third-party search index of already
        -public, already-crawled pages - it does not scrape or automate
        interaction with the platforms themselves, and coverage will be
        partial (many platforms block or limit crawler indexing).
        """
        if not self.social_domains:
            return {"context": []}
        if not state.get("messages"):
            return {"context": ["No conversation history available to search."]}
 
        query = self._resolve_search_query(state, call_label="search_query:social")
        if query.lower() in ("", "none", "null"):
            return {"context": ["Search skipped: no relevant search query needed."]}
 
        try:
            data = self.search_tool.invoke({"query": query, "include_domains": self.social_domains})
            search_docs = data.get("results", data) if isinstance(data, dict) else data
        except Exception as exc:
            logger.warning("Social-platform search failed: %s", exc)
            return {"context": [f"Social-platform search failed for query '{query}'."]}
 
        if not search_docs:
            logger.info(
                "No indexed results found on %s for query: %r", self.social_domains, query[:200]
            )
 
        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        )
        return {"context": [formatted_docs]}

    def _generate_answer(self, state: InterviewState) -> dict:
        """The expert answers the analyst's latest question using gathered context."""
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]

        system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
        answer = self._invoke(
            [SystemMessage(content=system_message)] + messages, call_label="generate_answer"
        )
        answer.name = "Expert"
        return {"messages": [answer]}

    @staticmethod
    def _save_interview(state: InterviewState) -> dict:
        """Flatten the message history into a plain-text transcript."""
        return {"interview": get_buffer_string(state["messages"])}

    def _route_messages(self, state: InterviewState, name: str = "Expert"):
        """Decide whether to continue the interview or wrap it up."""
        messages = state["messages"]
        max_num_turns = state.get("max_num_turns", self.max_interview_turns)

        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return "save_interview"

        if len(messages) >= 2 and INTERVIEW_CLOSING_LINE in (messages[-2].content or ""):
            return "save_interview"
        return "ask_question"

    def _write_section(self, state: InterviewState) -> dict:
        """Condense the interview context into a report section."""
        context = state["context"]
        analyst = state["analyst"]

        system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
        section = self._invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content=f"Use this source to write your section: {context}")],
            call_label="write_section",
        )
        return {"sections": [section.content]}

    def _build_interview_subgraph(self) -> StateGraph:
        """Build (but do not compile) the per-analyst interview graph."""
        builder = StateGraph(InterviewState)
        builder.add_node("ask_question", self._generate_question)
        builder.add_node("search_web", self._search_web)
        builder.add_node("search_wiki", self._search_wikipedia)
        builder.add_node("answer_question", self._generate_answer)
        builder.add_node("save_interview", self._save_interview)
        builder.add_node("write_section", self._write_section)

        builder.add_edge(START, "ask_question")
        builder.add_edge("ask_question", "search_web")
        builder.add_edge("ask_question", "search_wiki")
        builder.add_edge("search_web", "answer_question")
        builder.add_edge("search_wiki", "answer_question")

        if self.social_domains:
            builder.add_node("search_social", self._search_social)
            builder.add_edge("ask_question", "search_social")
            builder.add_edge("search_social", "answer_question")

        builder.add_conditional_edges(
            "answer_question", self._route_messages, ["ask_question", "save_interview"]
        )
        builder.add_edge("save_interview", "write_section")
        builder.add_edge("write_section", END)
        return builder

    def compile_standalone_interview_graph(self):
        """Compile the interview sub-graph with its own checkpointer, for
        running/testing a single analyst's interview in isolation."""
        return self._interview_builder.compile(checkpointer=MemorySaver()).with_config(
            run_name="Conduct Interview"
        )

    # Report writing

    def _write_report(self, state: ResearchGraphState) -> dict:
        sections = state["sections"]
        topic = state["topic"]
        formatted_sections = "\n\n".join(sections)

        system_message = REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_sections)
        report = self._invoke(
            [SystemMessage(content=system_message)]
            + [HumanMessage(content="Write a report based on these memos.")],
            call_label="write_report",
        )
        return {"content": report.content}

    def _write_introduction(self, state: ResearchGraphState) -> dict:
        formatted_sections = "\n\n".join(state["sections"])
        instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
            topic=state["topic"], formatted_str_sections=formatted_sections
        )
        intro = self._invoke(
            [SystemMessage(content=instructions)]
            + [HumanMessage(content="Write the report introduction.")],
            call_label="write_introduction",
        )
        return {"introduction": intro.content}

    def _write_conclusion(self, state: ResearchGraphState) -> dict:
        formatted_sections = "\n\n".join(state["sections"])
        instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
            topic=state["topic"], formatted_str_sections=formatted_sections
        )
        conclusion = self._invoke(
            [SystemMessage(content=instructions)]
            + [HumanMessage(content="Write the report conclusion.")],
            call_label="write_conclusion",
        )
        return {"conclusion": conclusion.content}

    @staticmethod
    def _finalize_report(state: ResearchGraphState) -> dict:
        """Reduce step: stitch introduction + body + conclusion together."""
        content = state["content"]
        if content.startswith("## Insights"):
            content = content.removeprefix("## Insights").lstrip()

        sources = None
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except ValueError:
                sources = None

        final_report = (
            state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
        )
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        return {"final_report": final_report}

    def _build_research_graph(self):
        """Build and compile the full end-to-end research graph."""
        builder = StateGraph(ResearchGraphState)
        builder.add_node("generate_analysts", self._generate_analysts)
        builder.add_node("human_feedback", self._human_feedback)
        builder.add_node("conduct_interview", self._interview_builder.compile())
        builder.add_node("write_report", self._write_report)
        builder.add_node("write_introduction", self._write_introduction)
        builder.add_node("write_conclusion", self._write_conclusion)
        builder.add_node("finalize_report", self._finalize_report)

        builder.add_edge(START, "generate_analysts")
        builder.add_edge("generate_analysts", "human_feedback")
        builder.add_conditional_edges(
            "human_feedback", self._route_after_feedback, ["generate_analysts", "conduct_interview"]
        )
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(
            ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
        )
        builder.add_edge("finalize_report", END)

        return builder.compile(interrupt_before=["human_feedback"], checkpointer=self.checkpointer)

    def start(self, topic: str, max_analysts: int, thread_id: str) -> List[Analyst]:
        """Run upto the first human-feedback interrupt."""
        thread = {"configurable": {"thread_id": thread_id}}
        analysts: List[Analyst] = []
        for event in self.graph.stream(
            {"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"
        ):
            if event.get("analysts"):
                analysts = event["analysts"]
        return analysts

    def submit_feedback(self, thread_id: str, feedback: Optional[str]) -> List[Analyst]:
        """Apply human feedback (or None to proceed) and resume until the
        next interrupt or run completion."""
        thread = {"configurable": {"thread_id": thread_id}}
        self.graph.update_state(
            thread, {"human_analyst_feedback": feedback}, as_node="human_feedback"
        )
        analysts: List[Analyst] = []
        for event in self.graph.stream(None, thread, stream_mode="values"):
            if event.get("analysts"):
                analysts = event["analysts"]
        return analysts

    def run_to_completion(self, thread_id: str) -> str:
        """Resume execution (after feedback has been finalized) through to
        the final report, logging node transitions as they happen."""
        thread = {"configurable": {"thread_id": thread_id}}
        for event in self.graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            logger.info("Completed node: %s", node_name)

        final_state = self.graph.get_state(thread)
        return final_state.values.get("final_report", "")

def print_analysts(analysts: List[Analyst]) -> None:
    for analyst in analysts:
        print(f"Name: {analyst.name}")
        print(f"Affiliation: {analyst.affiliation}")
        print(f"Role: {analyst.role}")
        print(f"Description: {analyst.description}")
        print("-" * 50)


def collect_human_feedback() -> Optional[str]:
    """Prompt the user once for analyst-panel feedback. Blank = proceed."""
    feedback = input(
        "\nEnter feedback to refine the analyst panel, or press Enter to proceed: "
    ).strip()
    return feedback or None


def save_report(report: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Final report written to %s", output_path)


def build_assistant(config: AppConfig) -> ResearchAssistant:
    """Construct the LLM, search tool, and assistant from config."""
    llm = ChatOllama(model=config.model_name, temperature=config.temperature)
    search_tool = TavilySearch(max_results=config.tavily_max_results)
    return ResearchAssistant(
        llm=llm,
        search_tool=search_tool,
        max_interview_turns=config.max_interview_turns,
        enable_social_search=config.enable_social_search,
        social_domains=config.social_domains
    )


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Multi-agent research assistant")
    parser.add_argument(
        "--topic",
        required=False,
        default=None,
        help="Research topic to investigate. Prompted for interactively if omitted.",
    )
    parser.add_argument("--max-analysts", type=int, default=3)
    parser.add_argument("--max-interview-turns", type=int, default=2)
    parser.add_argument("--model", default="qwen2.5:14b", dest="model_name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thread-id", default="1")
    parser.add_argument("--output", default="research_report.md", dest="output_path")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip the human feedback prompt and proceed straight through.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    parser.add_argument(
        "--log-file",
        default="research_agent.log",
        dest="log_file",
        help="Path to write logs to, in addition to the console. Pass an empty "
        "string ('') to disable file logging.",
    )
    parser.add_argument(
        "--social-domains",
        default="linkedin.com",
        help="Comma-separated domains to search for public, indexed employee/company "
        "social content (e.g. 'linkedin.com,x.com'). This queries Tavily's search "
        "index of already-public pages - it does not scrape these platforms directly.",
    )

    parser.add_argument(
        "--no-social-search",
        action="store_true",
        help="Disable the social-platform search node entirely.",
    )
    
    args = parser.parse_args()

    topic = args.topic
    if not topic:
        topic = input("Enter the research topic: ").strip()
        if not topic:
            parser.error("--topic is required (or must be entered when prompted).")

    social_domains = [d.strip() for d in args.social_domains.split(",") if d.strip()]

    return AppConfig(
        topic=topic,
        max_analysts=args.max_analysts,
        max_interview_turns=args.max_interview_turns,
        model_name=args.model_name,
        temperature=args.temperature,
        thread_id=args.thread_id,
        output_path=args.output_path,
        interactive_feedback=not args.non_interactive,
        social_domains=social_domains,
        enable_social_search=not args.no_social_search,
        log_file=args.log_file or None,
        verbose=args.verbose,
    )

def main() -> None:
    config = parse_args()
    configure_logging(verbose=config.verbose, log_file=config.log_file)
    log_gpu_availability()
    load_environment()

    assistant = build_assistant(config)

    logger.info("Generating analyst panel for topic: %s", config.topic)
    analysts = assistant.start(config.topic, config.max_analysts, config.thread_id)
    print_analysts(analysts)

    if config.interactive_feedback:
        while True:
            feedback = collect_human_feedback()
            analysts = assistant.submit_feedback(config.thread_id, feedback)
            if feedback is None:
                break
            print_analysts(analysts)
    else:
        assistant.submit_feedback(config.thread_id, None)

    logger.info("Running interviews and drafting the report...")
    report = assistant.run_to_completion(config.thread_id)
    assistant.log_token_summary()

    if not report:
        logger.error("No final report was produced.")
        return

    save_report(report, config.output_path)
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)


if __name__ == "__main__":
    main()