"""Deterministic quality checks for research assistant outputs.

These metrics check structure and consistency. They do not judge factual
accuracy or whether a source truly supports a claim; those require review.
"""

from __future__ import annotations

import re
import json
from itertools import combinations
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from privacy import redact_pii

_STOP_WORDS = {
    "about", "after", "also", "and", "are", "based", "being", "between",
    "by", "can", "description", "different", "each", "expert", "for",
    "from", "focus", "how", "into", "its", "more", "over", "role",
    "that", "the", "their", "these", "this", "through", "with", "within",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")


def _get(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOP_WORDS
    }


def evaluate_analyst_panel(analysts: Iterable[Any], expected_count: int) -> dict[str, Any]:
    """Return count, completeness, uniqueness, and focus-diversity metrics."""
    panel = list(analysts)
    names = [str(_get(analyst, "name")).strip().casefold() for analyst in panel]
    names = [name for name in names if name]
    identities = [
        str(_get(analyst, "affiliation")).strip().casefold()
        + "|"
        + str(_get(analyst, "role")).strip().casefold()
        for analyst in panel
    ]
    fields = ("name", "affiliation", "role", "description")
    complete = sum(
        all(str(_get(analyst, field)).strip() for field in fields)
        for analyst in panel
    )

    descriptions = [_tokens(str(_get(analyst, "description"))) for analyst in panel]
    pairwise_diversity = []
    for left, right in combinations(descriptions, 2):
        union = left | right
        pairwise_diversity.append(1.0 - len(left & right) / len(union) if union else 0.0)

    metrics = {
        "analyst_count": len(panel),
        "expected_count": expected_count,
        "count_matches": len(panel) == expected_count,
        "complete_analyst_ratio": complete / len(panel) if panel else 0.0,
        "unique_name_ratio": len(set(names)) / len(panel) if panel else 0.0,
        "unique_identity_ratio": len(set(identities)) / len(panel) if panel else 0.0,
        "focus_diversity": (
            sum(pairwise_diversity) / len(pairwise_diversity)
            if pairwise_diversity
            else 1.0 if len(panel) == 1 else 0.0
        ),
    }
    metrics["checks_pass"] = (
        metrics["count_matches"]
        and metrics["complete_analyst_ratio"] == 1.0
        and metrics["unique_name_ratio"] == 1.0
    )
    return metrics


def evaluate_report(report: str) -> dict[str, Any]:
    """Check report sections and whether numbered citations resolve to sources."""
    source_match = re.search(r"^##\s+Sources\s*$", report, re.IGNORECASE | re.MULTILINE)
    source_text = report[source_match.end():] if source_match else ""
    body = report[:source_match.start()] if source_match else report

    source_numbers = re.findall(r"^\[(\d+)\]\s+\S+", source_text, re.MULTILINE)
    cited_numbers = set(re.findall(r"\[(\d+)\]", body))
    defined_numbers = set(source_numbers)
    unresolved = sorted(cited_numbers - defined_numbers, key=int)
    unused = sorted(defined_numbers - cited_numbers, key=int)

    has_introduction = bool(re.search(r"^##\s+Introduction\s*$", report, re.I | re.M))
    has_conclusion = bool(re.search(r"^##\s+Conclusion\s*$", report, re.I | re.M))
    citations_resolve = bool(cited_numbers) and not unresolved and len(source_numbers) == len(defined_numbers)

    return {
        "has_introduction": has_introduction,
        "has_conclusion": has_conclusion,
        "has_sources_section": source_match is not None,
        "source_count": len(source_numbers),
        "unresolved_citations": unresolved,
        "unused_sources": unused,
        "citations_resolve": citations_resolve,
        "checks_pass": has_introduction and has_conclusion and source_match is not None and citations_resolve,
    }


class JudgeCriterion(BaseModel):
    score: int = Field(ge=1, le=5, description="Integer quality score from 1 (poor) to 5 (strong).")
    rationale: str = Field(description="Brief explanation grounded only in the supplied evidence.")
    concerns: list[str] = Field(default_factory=list, description="Specific unsupported or weak points, if any.")


class ReportJudgeEvaluation(BaseModel):
    faithfulness: JudgeCriterion
    relevance: JudgeCriterion
    citation_support: JudgeCriterion
    coverage: JudgeCriterion
    clarity: JudgeCriterion
    overall_score: int = Field(ge=1, le=5)
    summary: str


def _redact_strings(value: Any) -> tuple[Any, set[str]]:
    """Recursively mask recognized PII in evaluator output before returning it."""
    categories: set[str] = set()
    if isinstance(value, str):
        masked, found = redact_pii(value)
        return masked, set(found)
    if isinstance(value, list):
        masked_values = []
        for item in value:
            masked, found = _redact_strings(item)
            masked_values.append(masked)
            categories.update(found)
        return masked_values, categories
    if isinstance(value, dict):
        masked_dict = {}
        for key, item in value.items():
            masked, found = _redact_strings(item)
            masked_dict[key] = masked
            categories.update(found)
        return masked_dict, categories
    return value, categories


def evaluate_report_with_judge(
    judge_llm: Any,
    topic: str,
    report: str,
    evidence: Iterable[str],
    max_evidence_chars: int = 30000,
) -> dict[str, Any]:
    """Score a report against analyst memos using a structured LLM judge.

    Known PII patterns are masked before the judge request and in the returned
    rationales. The judge evaluates report consistency with analyst memos; it
    does not independently verify primary sources or factual truth.
    """
    safe_topic, topic_pii = redact_pii(topic)
    safe_report, report_pii = redact_pii(report)
    safe_evidence = []
    pii_categories = set(topic_pii + report_pii)
    for memo in evidence:
        safe_memo, found = redact_pii(str(memo))
        safe_evidence.append(safe_memo)
        pii_categories.update(found)

    if not safe_evidence:
        raise ValueError("LLM judge evaluation requires analyst memos as evidence.")

    evidence_text = "\n\n--- Analyst memo ---\n\n".join(safe_evidence)
    truncated = len(evidence_text) > max_evidence_chars
    if truncated:
        evidence_text = evidence_text[:max_evidence_chars]

    system_message = """You are a careful evaluator of a research report.
Score each criterion from 1 (poor) to 5 (strong), and explain each score briefly.
Use only the supplied topic, report, and analyst memos. Do not use outside
knowledge. Treat the analyst memos as the available evidence; they may
themselves contain errors, so do not claim this verifies primary-source truth.
For citation support, judge whether the report's citations and claims are
supported by the provided memos. List specific unsupported or weak claims in
concerns. Score relevance, coverage, and clarity separately from faithfulness.
"""
    payload = {
        "topic": safe_topic,
        "report": safe_report,
        "analyst_memos_evidence": evidence_text,
        "evidence_truncated": truncated,
    }
    structured = judge_llm.with_structured_output(
        ReportJudgeEvaluation,
        include_raw=True,
    )
    response = structured.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )

    if isinstance(response, dict):
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")
        if parsed is None:
            raise ValueError(f"LLM judge returned no parsed evaluation: {parsing_error}")
    else:
        parsed = response
    if not isinstance(parsed, ReportJudgeEvaluation):
        raise TypeError("LLM judge returned an unexpected evaluation format.")

    safe_result, output_pii = _redact_strings(parsed.model_dump(mode="json"))
    pii_categories.update(output_pii)
    safe_result["pii_redacted_categories"] = sorted(pii_categories)
    safe_result["evidence_truncated"] = truncated
    safe_result["judge_basis"] = "analyst_memos"
    return safe_result
