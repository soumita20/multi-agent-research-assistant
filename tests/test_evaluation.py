from unittest.mock import MagicMock

from evaluation import (
    JudgeCriterion,
    ReportJudgeEvaluation,
    evaluate_analyst_panel,
    evaluate_report,
    evaluate_report_with_judge,
)
from privacy import (
    anonymize_trace_payload,
    configure_langsmith_tracing,
    redact_pii,
)


def test_analyst_panel_reports_completeness_and_focus_diversity():
    analysts = [
        {
            "name": "Avery Chen",
            "affiliation": "University A",
            "role": "Energy economist",
            "description": "Studies grid pricing, electricity markets, and investment.",
        },
        {
            "name": "Morgan Lee",
            "affiliation": "Institute B",
            "role": "Climate scientist",
            "description": "Studies regional climate effects and extreme weather attribution.",
        },
    ]

    result = evaluate_analyst_panel(analysts, expected_count=2)

    assert result["checks_pass"] is True
    assert result["complete_analyst_ratio"] == 1.0
    assert result["unique_name_ratio"] == 1.0
    assert result["focus_diversity"] > 0.8


def test_analyst_panel_flags_duplicate_names_and_missing_fields():
    analyst = {
        "name": "Same Name",
        "affiliation": "Example",
        "role": "Researcher",
        "description": "Researches energy markets.",
    }
    incomplete = {**analyst, "name": "Same Name", "description": ""}

    result = evaluate_analyst_panel([analyst, incomplete], expected_count=2)

    assert result["checks_pass"] is False
    assert result["unique_name_ratio"] == 0.5
    assert result["complete_analyst_ratio"] == 0.5


def test_report_passes_when_sections_and_citations_are_consistent():
    report = """# Example

## Introduction
Findings support the claim [1].

## Conclusion
The finding is useful.

## Sources
[1] https://example.org/source
"""

    result = evaluate_report(report)

    assert result["checks_pass"] is True
    assert result["source_count"] == 1
    assert result["unresolved_citations"] == []


def test_report_flags_unresolved_citations_and_missing_sections():
    result = evaluate_report("## Introduction\nClaim [2].\n\n## Sources\n[1] source\n")

    assert result["checks_pass"] is False
    assert result["has_conclusion"] is False
    assert result["unresolved_citations"] == ["2"]


def test_pii_masker_redacts_email_phone_ssn_and_payment_card():
    text = (
        "Contact jane.doe@example.com or +1 (415) 555-2671. "
        "SSN 123-45-6789, card 4111 1111 1111 1111."
    )

    masked, categories = redact_pii(text)

    assert "jane.doe@example.com" not in masked
    assert "+1 (415) 555-2671" not in masked
    assert "123-45-6789" not in masked
    assert "4111 1111 1111 1111" not in masked
    assert set(categories) == {"EMAIL", "PHONE", "US_SSN", "PAYMENT_CARD"}


def test_pii_masker_leaves_non_pii_card_like_numbers_unchanged():
    text = "Order reference 4111 1111 1111 1112 is not a valid payment card."

    masked, categories = redact_pii(text)

    assert masked == text
    assert categories == []


def test_trace_anonymizer_masks_nested_inputs_outputs_and_metadata():
    trace = {
        "inputs": {"messages": [{"content": "Contact jane.doe@example.com"}]},
        "outputs": {"report": "Call +1 (415) 555-2671"},
        "metadata": {"note": "SSN 123-45-6789"},
    }

    masked = anonymize_trace_payload(trace)

    assert "jane.doe@example.com" not in str(masked)
    assert "+1 (415) 555-2671" not in str(masked)
    assert "123-45-6789" not in str(masked)
    assert "[REDACTED:EMAIL]" in str(masked)


def test_langsmith_tracing_uses_anonymizer_when_enabled(monkeypatch):
    import langsmith

    from privacy import anonymize_trace_payload

    configured = {}
    fake_client = object()
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(langsmith, "Client", lambda **kwargs: configured.update(kwargs) or fake_client)
    monkeypatch.setattr(langsmith, "configure", lambda **kwargs: configured.update(kwargs))

    configure_langsmith_tracing(True)

    assert configured["client"] is fake_client
    assert configured["enabled"] is True
    assert configured["anonymizer"] is anonymize_trace_payload


def test_llm_judge_masks_inputs_and_explanations():
    criterion = JudgeCriterion(
        score=4,
        rationale="The memo supports the finding; contact jane.doe@example.com was omitted.",
        concerns=[],
    )
    evaluation = ReportJudgeEvaluation(
        faithfulness=criterion,
        relevance=criterion,
        citation_support=criterion,
        coverage=criterion,
        clarity=criterion,
        overall_score=4,
        summary="Good report.",
    )
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": evaluation,
        "parsing_error": None,
    }

    result = evaluate_report_with_judge(
        judge_llm=llm,
        topic="Research contact jane.doe@example.com",
        report="Report for jane.doe@example.com",
        evidence=["Source memo for jane.doe@example.com"],
    )

    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert all("jane.doe@example.com" not in message.content for message in messages)
    assert "jane.doe@example.com" not in result["faithfulness"]["rationale"]
    assert result["pii_redacted_categories"] == ["EMAIL"]
    assert result["judge_basis"] == "analyst_memos"


def test_llm_judge_requires_evidence():
    llm = MagicMock()

    try:
        evaluate_report_with_judge(llm, "topic", "report", evidence=[])
    except ValueError as exc:
        assert "requires analyst memos" in str(exc)
    else:
        raise AssertionError("Expected evidence requirement to fail closed")

    llm.with_structured_output.assert_not_called()
