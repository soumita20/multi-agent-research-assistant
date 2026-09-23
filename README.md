# Multi-Agent Agentic AI Research Platform

A **stateful, multi-agent Agentic AI research platform built with LangGraph** that autonomously decomposes a research topic into specialized perspectives, conducts multi-turn expert interviews, retrieves evidence from multiple sources, and synthesizes the findings into a structured research report.

The project demonstrates production-oriented patterns for **Agentic AI orchestration, multi-agent systems, human-in-the-loop workflows, tool-augmented reasoning, dynamic parallelism, checkpointing, and observable stateful execution**.

---

## Architecture Overview

The platform follows a hierarchical multi-agent architecture in which a coordinator dynamically creates specialized analyst personas and orchestrates their execution.

```text
                         ┌─────────────────────┐
                         │    Research Topic   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Analyst Generation  │
                         │ Dynamic Personas    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Human-in-the-Loop   │
                         │ Review / Refinement │
                         └──────────┬──────────┘
                                    │
                             Dynamic Fan-out
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
       │  Analyst 1  │      │  Analyst 2  │      │  Analyst N  │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              │                    │                    │
              ▼                    ▼                    ▼
       Multi-turn Expert     Multi-turn Expert    Multi-turn Expert
          Interview             Interview             Interview
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Parallel Evidence       │
                    │ Retrieval               │
                    │                          │
                    │ • Tavily Web Search     │
                    │ • Wikipedia             │
                    │ • Social Search         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Section Generation       │
                    │ One section / analyst    │
                    └────────────┬─────────────┘
                                 │
                            Map-Reduce
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Report Synthesis         │
                    │                          │
                    │ Introduction             │
                    │ Insights                 │
                    │ Conclusion               │
                    │ Sources                  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Final Report  │
                         └───────────────┘
```

---

## Key Capabilities

### Multi-Agent Orchestration

The system dynamically generates a panel of specialized AI analysts based on the research topic.

Each analyst receives a distinct focus area and conducts an independent investigation.

This enables the system to approach complex topics from multiple perspectives rather than relying on a single LLM execution path.

### Dynamic Parallelism

Analyst workflows are dynamically created using LangGraph's `Send` API.

```python
Send(
    "conduct_interview",
    {
        "analyst": analyst,
        "interview_topic": topic,
        "messages": [...]
    }
)
```

This creates a scalable fan-out/fan-in execution model:

```text
Research Topic
      │
      ▼
Generate Analysts
      │
      ├──── Analyst A ────┐
      ├──── Analyst B ────┤
      ├──── Analyst C ────┤
      └──── Analyst N ────┘
                           │
                           ▼
                     Aggregate Results
```

---

## Human-in-the-Loop

The architecture includes an explicit human review checkpoint after analyst generation.

```text
Generate Analysts
       │
       ▼
   Human Review
       │
   ┌───┴────┐
   │        │
Feedback   Approve
   │        │
   ▼        ▼
Regenerate  Continue
```

LangGraph checkpointing and interrupts allow execution to pause before the research phase, enabling a human to review and refine the generated analyst panel.

This pattern demonstrates how human oversight can be incorporated into autonomous agent workflows without requiring the workflow itself to be rewritten.

---

## Stateful Agent Execution

The workflow uses typed state objects to maintain execution context across nodes.

### Research State

```python
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
```

### Interview State

Each analyst interview maintains its own message history, context, persona, and interview topic.

This provides isolation between parallel analyst workflows while allowing their outputs to be aggregated into the parent research graph.

---

## Tool-Augmented Agents

The expert agents do not rely exclusively on model knowledge.

For each interview turn, the system can retrieve supporting evidence from multiple external sources.

### Web Search

Uses **Tavily** to generate search context based on the current interview conversation.

### Wikipedia

Uses `WikipediaLoader` to retrieve relevant reference material.

### Social Search

Supports domain-scoped search for publicly indexed content through Tavily.

For example:

```text
linkedin.com
```

The implementation queries the search index rather than directly automating or scraping the social platform.

---

## Iterative Agent Interviews

Each analyst conducts a multi-turn interview with an expert agent.

```text
Analyst
   │
   ▼
Generate Question
   │
   ▼
Retrieve Evidence
   │
   ▼
Expert Answer
   │
   ▼
Continue?
 ┌─┴─┐
 │   │
Yes  No
 │   │
 └───┘
   │
   ▼
Save Interview
```

The analyst can continue asking questions until either:

- the configured interview-turn limit is reached, or
- the expert produces the defined closing response.

This provides a lightweight iterative reasoning loop while keeping the workflow deterministic and controllable.

---

## Evidence-Grounded Generation

Retrieved documents are passed into the expert agent as explicit context.

The answer-generation prompt instructs the model to:

- use only retrieved context;
- preserve source references;
- associate claims with source numbers;
- provide a consolidated source list.

This creates an evidence-oriented research workflow rather than an unconstrained generative response.

---

## Map-Reduce Report Synthesis

Once all analyst interviews are complete, each analyst produces an independent research section.

The system then performs a map-reduce style synthesis:

```text
              Analyst Sections
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Section A     Section B     Section C
       │             │             │
       └─────────────┼─────────────┘
                     ▼
              Report Synthesis
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Introduction  Insights  Conclusion
          │          │          │
          └──────────┼──────────┘
                     ▼
                Final Report
```

The final report preserves source references and produces a consolidated source section.

---

## LangGraph Architecture

The application is composed of a parent research graph and a reusable interview subgraph.

### Research Graph

```text
START
  │
  ▼
generate_analysts
  │
  ▼
human_feedback
  │
  ├──────────────► generate_analysts
  │
  ▼
conduct_interview
  │
  ├──────────────► write_report
  ├──────────────► write_introduction
  └──────────────► write_conclusion
                         │
                         ▼
                  finalize_report
                         │
                        END
```

### Interview Subgraph

```text
START
  │
  ▼
ask_question
  │
  ├──────────────► search_web
  ├──────────────► search_wiki
  └──────────────► search_social
                         │
                         ▼
                  answer_question
                         │
                    ┌────┴────┐
                    │         │
              ask_question  save_interview
                                │
                                ▼
                           write_section
                                │
                               END
```

---

## Checkpointing

The graph uses LangGraph checkpointing to maintain execution state.

```python
builder.compile(
    interrupt_before=["human_feedback"],
    checkpointer=self.checkpointer
)
```

A thread identifier is used to associate execution state with a specific research run.

This enables workflows to pause and resume without restarting the complete pipeline.

---

## Observability

The project includes application-level logging and token usage tracking.

Token usage is collected from:

- standard LangChain `usage_metadata`;
- Ollama response metadata where available.

The tracker maintains:

```text
Input Tokens
Output Tokens
Total Tokens
Number of LLM Calls
Per-call Usage
```

This provides visibility into the cost and execution characteristics of individual agent workflows.

---

## LangGraph Studio

The graph is exposed as a module-level graph for LangGraph Studio.

```python
graph = build_example_graph()
```

The intended project structure is:

```text
project/
├── research_agent.py
├── langgraph.json
├── requirements.txt
├── .env
└── README.md
```

Example `langgraph.json`:

```json
{
  "dependencies": [
    "."
  ],
  "graphs": {
    "research_agent": "./research_agent.py:graph"
  },
  "env": ".env"
}
```

This allows the workflow to be inspected and executed through LangGraph Studio while retaining the Python CLI interface.

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

## 2. Create a Python Virtual Environment

Python **3.10+** is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install and Start Ollama

This project uses **Ollama** for local LLM execution.

Install Ollama from:

https://ollama.com/

Download the default model:

```bash
ollama pull qwen2.5:14b
```

Verify that the model is available:

```bash
ollama list
```

The model can be changed through the `MODEL_NAME` configuration.

## 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=<your-tavily-api-key>
LANGSMITH_API_KEY=<your-langsmith-api-key>

MODEL_NAME=qwen2.5:14b
TEMPERATURE=0.0
MAX_INTERVIEW_TURNS=2
TAVILY_MAX_RESULTS=3
```

### Required

`TAVILY_API_KEY` is required for web search.

### Optional

`LANGSMITH_API_KEY` is optional and can be used for tracing and observability.

> Never commit your `.env` file or API keys to source control.

## 6. Run the Research Agent

```bash
python research_agent.py \
  --topic "Impact of AI agents on software engineering" \
  --max-analysts 3 \
  --output report.md
```

The generated report will be written to:

```text
report.md
```

## 7. Run with LangGraph Studio

Make sure `langgraph.json` is present in the project root:

```json
{
  "dependencies": [
    "."
  ],
  "graphs": {
    "research_agent": "./research_agent.py:graph"
  },
  "env": ".env"
}
```

Start the LangGraph development server:

```bash
langgraph dev
```

Open the URL displayed by the command to launch the LangGraph development environment.

From Studio you can inspect:

- Graph topology
- Node execution
- State transitions
- Agent execution
- Tool calls
- Human-in-the-loop interruptions
- Checkpointed execution

---

## Quick Architecture Test

For a lightweight end-to-end test:

```bash
python research_agent.py \
  --topic "Applications of Agentic AI in enterprise software" \
  --max-analysts 2 \
  --max-interview-turns 1
```

Using fewer analysts and interview turns is useful for validating the environment before running larger research workflows.

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| Local LLM | Ollama |
| Search | Tavily |
| Knowledge Retrieval | Wikipedia |
| Data Validation | Pydantic |
| Workflow State | TypedDict / LangGraph State |
| Persistence | LangGraph Checkpointing |
| Observability | Logging + Token Tracking |
| Development UI | LangGraph Studio |

---

## Project Structure

```text
.
├── research_agent.py       # Agent orchestration and graph definition
├── langgraph.json          # LangGraph Studio configuration
├── requirements.txt        # Python dependencies
├── .env                    # API keys / runtime configuration
└── README.md
```

---

## Configuration

The application supports configuration for:

```text
MODEL_NAME
TEMPERATURE
MAX_INTERVIEW_TURNS
TAVILY_MAX_RESULTS
TAVILY_API_KEY
LANGSMITH_API_KEY
```

Example:

```env
TAVILY_API_KEY=<your-tavily-key>
LANGSMITH_API_KEY=<your-langsmith-key>

MODEL_NAME=qwen2.5:14b
TEMPERATURE=0.0
MAX_INTERVIEW_TURNS=2
TAVILY_MAX_RESULTS=3
```

---

## Architectural Patterns Demonstrated

This project demonstrates several patterns relevant to enterprise Agentic AI platforms:

- **Multi-Agent Orchestration**
- **Coordinator / Supervisor Pattern**
- **Dynamic Agent Creation**
- **Stateful Agent Workflows**
- **Human-in-the-Loop**
- **Tool-Augmented Agents**
- **Iterative Agent Loops**
- **Dynamic Parallel Execution**
- **Map-Reduce Agent Architecture**
- **Checkpointed Execution**
- **Conditional Routing**
- **Evidence-Grounded Generation**
- **Observability and Token Tracking**
- **Local / Private LLM Integration**
- **LangGraph Studio Integration**

---

## Why This Architecture?

Traditional LLM applications generally follow:

```text
User → Prompt → LLM → Response
```

This project explores a more scalable Agentic AI architecture:

```text
User
 │
 ▼
Orchestrator
 │
 ├── Specialized Agent
 │      ├── Tools
 │      ├── Retrieval
 │      └── Iterative Reasoning
 │
 ├── Specialized Agent
 │      ├── Tools
 │      ├── Retrieval
 │      └── Iterative Reasoning
 │
 └── Specialized Agent
        ├── Tools
        ├── Retrieval
        └── Iterative Reasoning
 │
 ▼
Aggregation / Synthesis
 │
 ▼
Structured Output
```

The architecture separates **planning, specialized execution, evidence retrieval, synthesis, and human oversight**, providing a foundation that can be extended toward enterprise-grade Agentic AI systems.

---

## Future Extensions

Potential extensions include:

- Persistent production-grade checkpoint storage
- Additional retrieval and enterprise data connectors
- Agent-level access control
- Human approval at additional workflow stages
- Structured evaluation of agent outputs
- Agent performance and quality metrics
- Distributed execution for large analyst populations
- Model routing and fallback strategies
- Production tracing and evaluation
- API-based deployment
- Containerized deployment
- Additional Agentic AI workflows beyond research

---

## Author

Soumita Chowdhury