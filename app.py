
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from research_agent import AppConfig, build_assistant
from privacy import configure_langsmith_tracing, redact_pii
from evaluation import evaluate_report_with_judge
from app_settings import PROJECT_SETTINGS, provider_for_driver


load_dotenv()

st.set_page_config(
    page_title=PROJECT_SETTINGS["ui"]["page_title"],
    page_icon=PROJECT_SETTINGS["ui"]["page_icon"],
    layout=PROJECT_SETTINGS["ui"]["layout"],
)

st.title(PROJECT_SETTINGS["ui"]["title"])
st.caption(PROJECT_SETTINGS["ui"]["caption"])

def get_setting(name: str, default: str = "") -> str:
    """Read configuration from environment or Streamlit secrets."""
    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        # Local execution may not have a secrets.toml file.
        return default


def default_analyst_count() -> int:
    runtime = PROJECT_SETTINGS["runtime"]
    requested = int(get_setting("MAX_ANALYSTS", str(runtime["default_max_analysts"])))
    return min(max(requested, int(runtime["minimum_max_analysts"])), int(runtime["maximum_max_analysts"]))


def configure_privacy_safe_tracing() -> tuple[bool, str | None]:
    """Make tracing explicit and apply the local PII anonymizer when enabled."""
    langsmith_key = get_setting("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key

    enabled = get_setting("LANGSMITH_TRACING", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    try:
        configure_langsmith_tracing(enabled)
        return enabled, None
    except Exception as exc:
        configure_langsmith_tracing(False)
        return False, str(exc)


LANGSMITH_TRACING_ACTIVE, LANGSMITH_TRACING_ERROR = configure_privacy_safe_tracing()


def default_model_family() -> str:
    configured_driver = get_setting(
        "LLM_PROVIDER",
        PROJECT_SETTINGS["providers"][PROJECT_SETTINGS["runtime"]["default_family"]]["driver"],
    ).lower().strip()
    return provider_for_driver(PROJECT_SETTINGS, configured_driver) or PROJECT_SETTINGS["runtime"]["default_family"]


def available_models(family: str) -> dict[str, str]:
    """Return model IDs mapped to UI labels, retaining configured custom IDs."""
    provider = PROJECT_SETTINGS["providers"][family]
    models = {entry["id"]: entry["label"] for entry in provider["models"]}
    configured = get_setting(provider["model_setting"], provider["default_model"])
    models.setdefault(configured, f"{configured} (configured)")
    return models


def default_model_name(family: str) -> str:
    provider = PROJECT_SETTINGS["providers"][family]
    return get_setting(provider["model_setting"], provider["default_model"])


def configure_model(family: str, model: str):
    """Resolve the selected family and model using the project catalog."""
    try:
        provider = PROJECT_SETTINGS["providers"][family]
    except KeyError as exc:
        raise ValueError(f"Model family {family!r} is not configured.") from exc

    driver = provider["driver"]
    api_key = next(
        (get_setting(name) for name in provider.get("api_key_settings", []) if get_setting(name)),
        None,
    )
    base_url = ""
    mode = driver
    if driver == "ollama":
        mode = get_setting(provider["mode_setting"], provider["default_mode"]).lower().strip()
        base_url_key = provider.get("base_url_settings", {}).get(mode)
        if not base_url_key:
            raise ValueError(f"Ollama mode {mode!r} is not configured for model family {family!r}.")
        base_url = get_setting(base_url_key, provider["default_base_urls"][mode])
        if mode == "cloud" and not api_key:
            raise ValueError(f"One of {provider['api_key_settings']} is required for Ollama Cloud.")
        if mode == "cloud":
            os.environ[provider["api_key_settings"][0]] = api_key
        os.environ[base_url_key] = base_url
        os.environ[provider["model_setting"]] = model
    elif not api_key:
        raise ValueError(
            f"One of {provider.get('api_key_settings', [])} is required for {family}. "
            "Configure it in environment variables or Streamlit secrets."
        )
    return driver, base_url, model, api_key

# Session state

def initialise_state():
    runtime = PROJECT_SETTINGS["runtime"]
    defaults = {
        "stage": "input",
        "assistant": None,
        "thread_id": None,
        "topic": "",
        "max_analysts": default_analyst_count(),
        "analysts": [],
        "report": "",
        "error": None,
        "ollama_mode": None,
        "pii_warning": None,
        "judge_evaluation": None,
        "judge_evaluation_error": None,
        "model_family": default_model_family(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    for key in (
        "stage",
        "assistant",
        "thread_id",
        "topic",
        "max_analysts",
        "analysts",
        "report",
        "error",
        "ollama_mode",
        "pii_warning",
        "judge_evaluation",
        "judge_evaluation_error",
    ):
        st.session_state.pop(key, None)


def analyst_to_dict(analyst):
    if hasattr(analyst, "model_dump"):
        return analyst.model_dump()

    if hasattr(analyst, "dict"):
        return analyst.dict()

    if isinstance(analyst, dict):
        return analyst

    return {"name": str(analyst)}


def display_analysts(analysts):
    st.subheader("Your research analysts")

    for index, analyst in enumerate(analysts, start=1):
        details = analyst_to_dict(analyst)

        name = details.get("name", f"Analyst {index}")

        with st.expander(f"{index}. {name}", expanded=True):
            st.write(
                f"**Role:** {details.get('role', 'Not specified')}"
            )
            st.write(
                "**Affiliation:** "
                f"{details.get('affiliation', 'Not specified')}"
            )
            st.write(
                "**Research focus:** "
                f"{details.get('description', 'Not specified')}"
            )

# Workflow operations

def generate_analysts(topic: str, max_analysts: int, model_family: str, model: str):
    mode, base_url, model, api_key = configure_model(model_family, model)
    runtime = PROJECT_SETTINGS["runtime"]

    tavily_key_setting = runtime["tavily_api_key_setting"]
    tavily_key = get_setting(tavily_key_setting)

    if not tavily_key:
        raise ValueError(
            f"{tavily_key_setting} is missing. Configure it in .env "
            "for local development or Streamlit secrets for cloud."
        )

    os.environ[runtime["tavily_client_env_var"]] = tavily_key


    config = AppConfig(
        topic=topic,
        max_analysts=max_analysts,
        max_interview_turns=int(get_setting("MAX_INTERVIEW_TURNS", str(runtime["max_interview_turns"]))),
        model_name=model,
        base_url=base_url,
        llm_provider=mode,
        api_key=api_key,
        temperature=float(get_setting("TEMPERATURE", str(runtime["temperature"]))),
        tavily_max_results=int(get_setting("TAVILY_MAX_RESULTS", str(runtime["tavily_max_results"]))),
        thread_id=uuid.uuid4().hex,
        enable_social_search=get_setting(
            "ENABLE_SOCIAL_SEARCH", str(runtime["enable_social_search"])
        ).lower() in {"1", "true", "yes", "on"},
        social_domains=[
            domain.strip()
            for domain in get_setting(
                "SOCIAL_DOMAINS", ",".join(runtime["social_domains"])
            ).split(",")
            if domain.strip()
        ],
    )

    assistant = build_assistant(config)

    analysts = assistant.start(
        topic=topic,
        max_analysts=max_analysts,
        thread_id=config.thread_id,
    )

    if not analysts:
        raise RuntimeError(
            "The workflow did not generate any analysts."
        )

    st.session_state.assistant = assistant
    st.session_state.thread_id = config.thread_id
    st.session_state.topic = topic
    st.session_state.max_analysts = max_analysts
    st.session_state.analysts = analysts
    st.session_state.ollama_mode = get_setting(
        PROJECT_SETTINGS["providers"][model_family].get("mode_setting", ""),
        PROJECT_SETTINGS["providers"][model_family].get("default_mode", mode),
    )
    st.session_state.report = ""
    st.session_state.error = None
    st.session_state.stage = "feedback_choice"


def apply_feedback(feedback: str):
    feedback, pii_categories = redact_pii(feedback)
    if pii_categories:
        st.session_state.pii_warning = (
            "Personal data was masked in your feedback before it was sent to the model. "
            f"Detected categories: {', '.join(pii_categories)}."
        )

    analysts = st.session_state.assistant.submit_feedback(
        st.session_state.thread_id,
        feedback,
    )

    if analysts:
        st.session_state.analysts = analysts

    st.session_state.error = None
    st.session_state.stage = "feedback_choice"



def execute_research():
    """Resume the graph and generate the final report."""
    status = None
    try:
        with st.status(
            "Research in progress...",
            expanded=True,
        ) as status:
            st.write(
                "Running analyst interviews and gathering evidence."
            )

            report = st.session_state.assistant.run_to_completion(
                st.session_state.thread_id
            )

            if not report or not str(report).strip():
                raise RuntimeError(
                    "Research finished without a final report."
                )

            st.session_state.report = str(report)
            masked_report, pii_categories = redact_pii(st.session_state.report)
            if pii_categories:
                st.session_state.report = masked_report
                st.session_state.pii_warning = (
                    "Personal data was masked in the generated report before display and download. "
                    f"Detected categories: {', '.join(pii_categories)}."
                )
            st.session_state.judge_evaluation = None
            st.session_state.judge_evaluation_error = None
            if get_setting("ENABLE_LLM_JUDGE", "false").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                try:
                    st.session_state.judge_evaluation = evaluate_report_with_judge(
                        judge_llm=st.session_state.assistant.llm,
                        topic=st.session_state.topic,
                        report=st.session_state.report,
                        evidence=st.session_state.assistant.last_evaluation_evidence,
                    )
                except Exception as judge_exc:
                    st.session_state.judge_evaluation_error = str(judge_exc)
            st.session_state.stage = "complete"
            st.session_state.error = None

            status.update(
                label="Research completed successfully.",
                state="complete",
                expanded=False,
            )

    except Exception as exc:
        st.session_state.error = f"{type(exc).__name__}: {exc}"
        st.session_state.stage = "research_error"
        if status is not None:
            status.update(
                label="Research stopped because an error occurred.",
                state="error",
                expanded=True,
            )

# Sidebar: inference configuration

initialise_state()

if st.session_state.pii_warning:
    st.warning(st.session_state.pii_warning)

with st.sidebar:
    st.header(PROJECT_SETTINGS["ui"]["sidebar_header"])
    model_family = st.selectbox(
        PROJECT_SETTINGS["ui"]["model_family_label"],
        list(PROJECT_SETTINGS["providers"]),
        key="model_family",
    )
    models = available_models(model_family)
    default_name = default_model_name(model_family)
    model_name = st.selectbox(
        PROJECT_SETTINGS["ui"]["model_name_label"],
        list(models),
        index=list(models).index(default_name),
        format_func=lambda model_id: models[model_id],
        key=f"model_name_{model_family.lower()}",
    )
    provider_settings = PROJECT_SETTINGS["providers"][model_family]
    if provider_settings["driver"] == "ollama":
        mode = get_setting(provider_settings["mode_setting"], provider_settings["default_mode"]).lower()
        st.write(f"**Selected provider:** {model_family} ({mode})")
        st.caption("Install local model IDs in your configured local model runtime before selecting them.")
    else:
        st.write(f"**Selected provider:** {model_family}")
        st.caption(provider_settings.get(
            "usage_note", "Availability and request limits depend on your provider account."
        ))

    st.caption(
        "Local mode requires Ollama to be running on the "
        "same machine as Streamlit."
    )
    st.caption(
        "LLM-as-judge evaluation is opt-in with ENABLE_LLM_JUDGE=true and "
        "uses the model selected above."
    )
    st.caption(
        "LangSmith tracing: "
        + ("enabled with recognized-PII masking" if LANGSMITH_TRACING_ACTIVE else "disabled")
    )
    if LANGSMITH_TRACING_ERROR:
        st.warning("LangSmith tracing was disabled: " + LANGSMITH_TRACING_ERROR)

# Stage 1: Topic and analyst count

if st.session_state.stage == "input":
    st.subheader("Start a research project")

    with st.form("research_input_form"):
        topic = st.text_area(
            PROJECT_SETTINGS["ui"]["research_input_label"],
            placeholder=PROJECT_SETTINGS["ui"]["research_input_placeholder"],
            height=120,
        )

        max_analysts = st.number_input(
            PROJECT_SETTINGS["ui"]["max_analysts_label"],
            min_value=int(PROJECT_SETTINGS["runtime"]["minimum_max_analysts"]),
            max_value=int(PROJECT_SETTINGS["runtime"]["maximum_max_analysts"]),
            value=default_analyst_count(),
            step=1,
        )

        submitted = st.form_submit_button(
            PROJECT_SETTINGS["ui"]["search_button_label"],
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not topic.strip():
            st.warning("Please enter a research topic.")
        else:
            try:
                safe_topic, pii_categories = redact_pii(topic.strip())
                if pii_categories:
                    st.session_state.pii_warning = (
                        "Personal data was masked in your topic before it was sent to the model. "
                        f"Detected categories: {', '.join(pii_categories)}."
                    )
                    st.warning(st.session_state.pii_warning)
                else:
                    st.session_state.pii_warning = None
                with st.spinner("Generating analyst panel..."):
                    generate_analysts(
                        safe_topic,
                        int(max_analysts),
                        model_family,
                        model_name,
                    )

                st.rerun()

            except Exception as exc:
                st.session_state.error = str(exc)

# Stage 2: Analyst review and human feedback

if st.session_state.stage in (
    "feedback_choice",
    "feedback_text",
):
    st.subheader("Review the analyst panel")
    st.write(f"**Research topic:** {st.session_state.topic}")

    display_analysts(st.session_state.analysts)

    st.divider()

    st.subheader("Would you like to provide more feedback?")
    st.write(
        "You can refine the analyst panel before research begins. "
        "Choose No if you are happy with the proposed analysts."
    )

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button(
            "Yes, I want to provide feedback",
            use_container_width=True,
        ):
            st.session_state.stage = "feedback_text"
            st.rerun()

    with col_no:
        if st.button(
            "No, continue with research",
            type="primary",
            use_container_width=True,
        ):
            try:
                st.session_state.assistant.submit_feedback(
                    st.session_state.thread_id,
                    None,
                )

                st.session_state.stage = "researching"
                st.session_state.error = None
                st.rerun()

            except Exception as exc:
                st.session_state.error = str(exc)


    if st.session_state.stage == "feedback_text":
        with st.form("human_feedback_form"):
            feedback = st.text_area(
                "Enter your feedback",
                placeholder=(
                    "Example: Add a cybersecurity analyst, "
                    "prioritize EU regulations, and compare "
                    "enterprise deployment strategies."
                ),
                height=140,
            )

            feedback_submitted = st.form_submit_button(
                "Apply feedback and regenerate analysts",
                type="primary",
                use_container_width=True,
            )

        if feedback_submitted:
            if not feedback.strip():
                st.warning("Please enter your feedback.")
            else:
                try:
                    _, pii_categories = redact_pii(feedback.strip())
                    if pii_categories:
                        st.session_state.pii_warning = (
                            "Personal data was masked in your feedback before it was sent to the model. "
                            f"Detected categories: {', '.join(pii_categories)}."
                        )
                        st.warning(st.session_state.pii_warning)
                    with st.spinner("Updating analyst perspectives..."):
                        apply_feedback(feedback.strip())

                    st.rerun()

                except Exception as exc:
                    st.session_state.error = str(exc)

# Stage 3: Research progress

if st.session_state.stage == "researching":
    st.header("Research in progress")

    st.write(f"**Topic:** {st.session_state.topic}")

    execute_research()

# Research error and retry

if st.session_state.stage == "research_error":
    st.error("Research stopped before the report could be completed.")
    with st.expander("Error details", expanded=True):
        st.code(st.session_state.error or "No error details were returned.")

    if st.button("Retry research", type="primary"):
        st.session_state.error = None
        st.session_state.stage = "researching"
        st.rerun()

    if st.button("Start a new project with a different model"):
        reset_session()
        st.rerun()

# Stage 4: Display errors

if (
    st.session_state.error
    and st.session_state.stage != "research_error"
):
    st.error("The requested action failed. See the details below.")
    with st.expander("Error details", expanded=True):
        st.code(st.session_state.error)

# Stage 5: Final report and download

if st.session_state.stage == "complete":
    st.divider()
    st.header("Final Research Report")

    research_warnings = getattr(st.session_state.assistant, "run_warnings", [])
    if research_warnings:
        st.warning(
            "The report was generated, but one or more research sources or model calls had issues. "
            "The report may have incomplete evidence."
        )
        with st.expander("Research warnings", expanded=True):
            for warning in research_warnings:
                st.write(f"- {warning}")

    if st.session_state.pii_warning:
        st.warning(st.session_state.pii_warning)

    st.markdown(st.session_state.report)

    if st.session_state.judge_evaluation_error:
        st.warning(
            "The report was generated, but its LLM judge evaluation failed: "
            + st.session_state.judge_evaluation_error
        )

    evaluation = st.session_state.judge_evaluation
    if evaluation:
        st.subheader("LLM judge evaluation")
        st.caption(
            "Scores compare the report with the analyst memos. They are review signals, "
            "not proof of factual accuracy or primary-source support."
        )
        st.metric("Overall", f"{evaluation['overall_score']} / 5")
        for criterion in (
            "faithfulness",
            "relevance",
            "citation_support",
            "coverage",
            "clarity",
        ):
            detail = evaluation[criterion]
            with st.expander(f"{criterion.replace('_', ' ').title()}: {detail['score']} / 5"):
                st.write(detail["rationale"])
                if detail["concerns"]:
                    st.write("Concerns:")
                    for concern in detail["concerns"]:
                        st.write(f"- {concern}")
        if evaluation["pii_redacted_categories"]:
            st.caption(
                "The judge input or explanations had recognized PII patterns masked: "
                + ", ".join(evaluation["pii_redacted_categories"])
            )

    st.download_button(
        label="Download research report (.md)",
        data=st.session_state.report,
        file_name="research_report.md",
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )

    if st.button("Start a new research project"):
        reset_session()
        st.rerun()
