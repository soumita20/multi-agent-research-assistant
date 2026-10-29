
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from research_agent import AppConfig, build_assistant


load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Multi-Agent Research Assistant")
st.caption(
    "Explore a topic with specialized AI analysts, "
    "human feedback, research interviews, and report synthesis."
)

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


def configure_ollama():
    """
    Configure local Ollama or Ollama Cloud.

    Local:
        OLLAMA_MODE=local
        OLLAMA_BASE_URL=http://localhost:11434
        OLLAMA_MODEL=qwen2.5:14b

    Cloud:
        OLLAMA_MODE=cloud
        OLLAMA_BASE_URL=https://ollama.com
        OLLAMA_MODEL=<supported cloud model>
        OLLAMA_API_KEY=<your Ollama API key>
    """
    mode = get_setting("OLLAMA_MODE", "local").lower().strip()

    model = get_setting(
        "OLLAMA_MODEL",
        "qwen2.5:14b",
    )

    if mode == "local":
        base_url = get_setting(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )

    elif mode == "cloud":
        base_url = get_setting(
            "OLLAMA_BASE_URL",
            "https://ollama.com",
        )

        api_key = get_setting("OLLAMA_API_KEY")

        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY is required for Ollama Cloud. "
                "Configure it in Streamlit secrets."
            )

        # ChatOllama does not automatically use this key for
        # direct requests to ollama.com. Configure its HTTP
        # clients to send the required Bearer token.
        os.environ["OLLAMA_API_KEY"] = api_key

    else:
        raise ValueError(
            "OLLAMA_MODE must be 'local' or 'cloud'."
        )

    os.environ["OLLAMA_BASE_URL"] = base_url
    os.environ["OLLAMA_MODEL"] = model

    return mode, base_url, model

# Session state

def initialise_state():
    defaults = {
        "stage": "input",
        "assistant": None,
        "thread_id": None,
        "topic": "",
        "max_analysts": int(get_setting("MAX_ANALYSTS", "3")),
        "analysts": [],
        "report": "",
        "error": None,
        "ollama_mode": None,
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

def generate_analysts(topic: str, max_analysts: int):
    mode, base_url, model = configure_ollama()

    tavily_key = get_setting("TAVILY_API_KEY")

    if not tavily_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. Configure it in .env "
            "for local development or Streamlit secrets for cloud."
        )

    os.environ["TAVILY_API_KEY"] = tavily_key


    config = AppConfig(
        topic=topic,
        max_analysts=max_analysts,
        max_interview_turns=2,
        model_name=model,
        base_url=base_url,
        temperature=0.0,
        tavily_max_results=int(get_setting("TAVILY_MAX_RESULTS", "3")),
        thread_id=uuid.uuid4().hex,
        enable_social_search=True,
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
    st.session_state.ollama_mode = mode
    st.session_state.report = ""
    st.session_state.error = None
    st.session_state.stage = "feedback_choice"


def apply_feedback(feedback: str):
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
            st.session_state.stage = "complete"
            st.session_state.error = None

            status.update(
                label="Research completed successfully.",
                state="complete",
                expanded=False,
            )

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.stage = "research_error"

# Sidebar: inference configuration

initialise_state()

with st.sidebar:
    st.header("Inference configuration")

    current_mode = get_setting("OLLAMA_MODE", "local").lower()

    st.write(
        "**Configured mode:** "
        + ("Ollama Cloud" if current_mode == "cloud" else "Local Ollama")
    )

    st.caption(
        "Change OLLAMA_MODE in your local .env file or "
        "Streamlit Cloud secrets."
    )

    st.caption(
        "Local mode requires Ollama to be running on the "
        "same machine as Streamlit."
    )

# Stage 1: Topic and analyst count

if st.session_state.stage == "input":
    st.subheader("Start a research project")

    with st.form("research_input_form"):
        topic = st.text_area(
            "What topic would you like to research?",
            placeholder=(
                "Example: Enterprise adoption of agentic AI "
                "and its security implications"
            ),
            height=120,
        )

        max_analysts = st.number_input(
            "Maximum number of analysts",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
        )

        submitted = st.form_submit_button(
            "Generate analysts",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not topic.strip():
            st.warning("Please enter a research topic.")
        else:
            try:
                with st.spinner("Generating analyst panel..."):
                    generate_analysts(
                        topic.strip(),
                        int(max_analysts),
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
    st.error(
        st.session_state.error or "Research could not be completed."
    )

    if st.button("Retry research", type="primary"):
        st.session_state.error = None
        st.session_state.stage = "researching"
        st.rerun()

# Stage 4: Display errors

if (
    st.session_state.error
    and st.session_state.stage != "research_error"
):
    st.error(st.session_state.error)

# Stage 5: Final report and download

if st.session_state.stage == "complete":
    st.divider()
    st.header("Final Research Report")

    st.markdown(st.session_state.report)

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
