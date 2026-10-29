# Research-assistant (Elisa)

## Purpose
Elisa is a research assistant agent that helps a user research any topic. She asks the user for a research topic (or takes it from the command line) and generates a panel of AI analyst personas, each covering a different sub-topic. She shows the panel to the user and asks for feedback. If the user gives feedback, she rebuilds the panel and shows it again. When the user accepts the panel, each analyst interviews an AI expert on their sub-topic. Before every answer, the expert gathers context from the web, Wikipedia and LinkedIn. Each interview becomes a report section. Elisa then merges all sections into one final report with an introduction, body, conclusion and sources, and saves it as a markdown file.

## Goals
- Give the user a sourced, multi-perspective report on any topic in one run.
- Keep the user in control of the research angle through the analyst feedback step.
- Make every claim in the report traceable to a source.

## Non-goals
- Not a source of verified or expert-reviewed advice (for example medical or legal).
- Does not scrape LinkedIn or any social platform directly.
- Does not run unattended research over long periods or remember past runs.

## Users
- A person who wants a quick first-draft research report on a topic and will check the sources themselves.

## Inputs
- Research topic (required). Asked interactively if not given.
- Optional settings: number of analysts (default 3), interview turns per analyst (default 2), model name (default qwen2.5:14b), temperature (default 0.0), output file path, thread id, social domains, verbose logging, log file.
- Flags: --non-interactive skips the feedback prompt. --no-social-search turns off LinkedIn search.
- Analyst feedback (optional): free text. Pressing Enter accepts the current panel.
- Report file name (optional): the user can type a file name after the report is generated. Pressing Enter accepts the name built from the topic. Not asked when --output or --non-interactive is used.

## Outputs
- A markdown report: title and introduction, body (insights), conclusion, and a numbered Sources list.
- Console print of the report and the analyst panel.
- A log file (default research_agent.log) with node progress and token usage per call and in total.
- Optional LangSmith traces when the API key is set.
- A markdown report named after the topic (for example, topic-name.md), or the path given with --output. Existing files are not overwritten when the name is generated from the topic.

## Workflow
1. Generate analysts: create the requested number of personas (name, role, affiliation, focus).
2. Human feedback: show the panel. If feedback is given, regenerate. If not, continue.
3. Interviews (one per analyst, run in parallel):
   - The analyst asks a question.
   - Web search, Wikipedia search and LinkedIn search run in parallel for that question.
   - The expert answers using only the retrieved context, with citations.
   - This repeats until the turn limit is reached or the analyst says the closing line.
4. Write a report section from each interview.
5. Write the body, introduction and conclusion in parallel.
6. Combine them into the final report and save it.

## Tools
- **Tavily Search** - Searches the web for information related to the analyst's latest question.
- **Wikipedia Search** - Loads up to 2 Wikipedia pages related to the question.
- **Social Search (LinkedIn)** - Uses Tavily limited to configured domains (default linkedin.com) to find public, search-indexed pages. Coverage is partial because many pages are not indexed.
- **LLM (Ollama, qwen2.5:14b)** - Generates analysts, questions, search queries, answers, sections and the final report.

## Behavioural rules
- Web, Wikipedia and social search run on every interview turn. The model does not choose whether to search.
- Social search runs only when it is enabled and at least one domain is configured.
- The search query is generated from the conversation, focused on the analyst's last question.
- If query generation fails or returns empty, fall back to the research topic, then the analyst's focus, then a short excerpt of the last message.
- The expert must use only the provided context and must not add outside facts or assumptions.
- The expert must cite sources as [1], [2] and list them at the end of the answer.
- Report sections must have a title, a summary of about 400 words or less, and a Sources list with no duplicates.
- The final report must not name analysts or interviewers.
- Sources in the final report must be listed in order and not repeated.

## Error handling
- If a search tool fails or returns nothing, the run continues. The expert gets a note that the search failed or returned no results.
- If analyst generation cannot be parsed, the run stops with an error.
- If no report is produced, an error is logged and no file is written.

## Scenarios and expected behaviour
| Input | Expected behaviour |
|-------|--------------------|
| How is the job market in Germany? | Elisa generates analysts for sub-topics such as demand by sector, salaries and hiring trends. On every turn she searches the web, Wikipedia and LinkedIn. The web results are likely the main source. Output is one sourced report. |
| What kind of employees are hired for Agentic AI in BMW? | Same flow. Social search is most useful here because LinkedIn profiles and posts show the qualifications of current BMW employees. Web and Wikipedia add company context. |
| User gives feedback such as "add an analyst on regulation" | The panel is regenerated using the feedback and shown again. Interviews start only after the user presses Enter. |
| Run with --non-interactive | The feedback step is skipped and the first panel is used. |
| Run with --no-social-search | Only web and Wikipedia are used. |
| Tavily fails or Wikipedia finds nothing | The run continues with the remaining sources and the gap is logged. |

## Requirements
- Python with LangGraph, LangChain, langchain-ollama, langchain-tavily, and the wikipedia package.
- Ollama running locally with the chosen model.
- TAVILY_API_KEY (required). LANGSMITH_API_KEY (optional, for tracing). Missing keys are asked for at startup.

## Known limitations and risks
- The expert is an LLM limited to retrieved context. Weak search results lead to weak or thin answers.
- Analyst names and affiliations are fictional.
- LinkedIn results may be few, outdated or empty, and may include personal information about individuals. Use them as signals, not facts.
- A 14B local model can make mistakes in structured output, citations and summaries.
- Two interview turns per analyst gives limited depth.
- Search is run on every turn even when not needed, which adds cost and time.

## Success metrics
- Report is produced in one run without manual fixes.
- Every section has at least one source, and cited links open correctly.
- Token usage and run time per report are logged and tracked.
- User accepts the panel in at most 2 feedback rounds.