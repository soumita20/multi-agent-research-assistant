# Evaluation strategy

The project uses three levels of evaluation:

1. **Unit tests** check deterministic behavior: provider construction, schema parsing, retry handling, token accounting, panel quality metrics, and report structure/citation consistency. These run offline with mocked model and search calls.
2. **Golden-topic runs** use a small, fixed set of research topics and save the generated analyst panel, source set, interviews, and report for comparison across model or prompt changes. Keep model name, prompt revision, date, and settings with each run.
3. **Human quality review** scores each run from 1–5 for topic coverage, analyst differentiation, source relevance/credibility, citation support, factual accuracy, and usefulness. Reviewers should open cited sources; deterministic citation matching alone does not establish that a citation supports a claim.

The helpers in `evaluation.py` report objective indicators, not an overall truth score. In particular, focus-word diversity is only a signal and must not substitute for topical coverage or expert review. Calibrate judge scores against human-reviewed examples and retain the rubric and examples with the results.

## Privacy checks

`privacy.redact_pii` runs locally before a topic or feedback is sent to the
research workflow, and on the final report before it is displayed or saved.
It masks common email addresses, phone numbers, US Social Security numbers,
and Luhn-valid payment-card numbers. The Streamlit app and CLI warn when a
mask is applied. This pattern-based pass does not detect names, street
addresses, or all sensitive text and must not be treated as a guarantee that
content is anonymous. Avoid entering highly sensitive information.

If LangSmith tracing is enabled, intermediate prompts and model outputs may
be traced. Tracing is disabled by default and must be explicitly enabled with
`LANGSMITH_TRACING=true`. When enabled, the LangSmith client anonymizer applies
the same recognized-pattern masking to serialized trace inputs, outputs,
metadata, and errors. Retrieved web or Wikipedia content may still contain
personal information, and this scanner misses names, addresses, and other
identifiers. Do not enable remote tracing or send research material to remote
providers when its privacy requirements do not permit that processing.

PII checks are deterministic privacy gates, not LLM-judge scores. Do not send
unmasked examples to a remote judge. The optional report judge is enabled with
`ENABLE_LLM_JUDGE=true` and uses the configured generation model/provider. It
scores faithfulness to analyst memos, relevance, citation support, coverage,
and clarity from 1–5 with brief rationales and concerns. It masks recognized
PII patterns in the topic, report, memos, and returned explanations. Its
faithfulness score only compares the report with the memos; it does not
independently verify the memos or open cited primary sources. Keep human review
for factual accuracy and citation support.

Run the automated tests with:

```powershell
uv sync --group dev
uv run pytest
```
