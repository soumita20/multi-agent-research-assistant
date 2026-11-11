from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def make_app():
    return AppTest.from_file(str(APP_PATH)).run()


def test_model_family_dropdown_filters_model_names(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    app = make_app()

    assert not app.exception
    assert [selectbox.label for selectbox in app.selectbox] == [
        "Model family",
        "Model name",
    ]
    assert app.selectbox[0].value == "Ollama"
    assert "Gemma 3 4B (local)" in app.selectbox[1].options

    app.selectbox[0].set_value("Groq").run()
    assert "GPT-OSS 20B (free tier)" in app.selectbox[1].options
    assert "Qwen 3.8 27B (free tier)" in app.selectbox[1].options

    app.selectbox[0].set_value("Gemini").run()
    assert "Gemini 3.8 Flash (free tier)" in app.selectbox[1].options


def test_search_button_requires_a_topic_before_starting(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    app = make_app()

    search_button = next(button for button in app.button if button.label == "Search")
    search_button.click().run()

    assert any("Please enter a research topic." in warning.value for warning in app.warning)
    assert not app.exception


def test_research_error_is_displayed_with_details(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    app = make_app()
    app.session_state["stage"] = "research_error"
    app.session_state["error"] = "GoogleRateLimitError: daily quota exceeded"
    app.run()

    assert any("Research stopped before the report" in error.value for error in app.error)
    assert any("GoogleRateLimitError: daily quota exceeded" in code.value for code in app.code)
    assert not app.exception


def test_completed_report_shows_recoverable_search_warnings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    app = make_app()
    app.session_state["stage"] = "complete"
    app.session_state["report"] = "# Research report"
    app.session_state["assistant"] = SimpleNamespace(
        run_warnings=["Web search authentication failed (HTTP 401). Check the configured search API key."]
    )
    app.run()

    assert any("incomplete evidence" in warning.value for warning in app.warning)
    assert any("authentication failed (HTTP 401)" in markdown.value for markdown in app.markdown)
    assert not app.exception
