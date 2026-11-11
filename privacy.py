"""Local, pattern-based detection and masking for common direct identifiers.

This lightweight scanner covers email addresses, phone numbers, US SSNs, and
payment-card numbers. It is not a general PII detector and does not identify
names, street addresses, or context-dependent sensitive information.
"""

from __future__ import annotations

import re
import os
from collections.abc import Mapping


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)"
)


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask common direct identifiers locally and return detected categories."""
    categories: list[str] = []

    def replace(pattern: re.Pattern[str], category: str, predicate=None) -> None:
        nonlocal text
        found = False

        def mask(match: re.Match[str]) -> str:
            nonlocal found
            if predicate is not None and not predicate(match.group(0)):
                return match.group(0)
            found = True
            return f"[REDACTED:{category}]"

        text = pattern.sub(mask, text)
        if found and category not in categories:
            categories.append(category)

    replace(_EMAIL, "EMAIL")
    replace(_SSN, "US_SSN")
    replace(_CARD, "PAYMENT_CARD", _passes_luhn)
    replace(_PHONE, "PHONE")
    return text, categories


def anonymize_trace_payload(value):
    """Recursively mask recognized identifiers in serialized trace data."""
    if isinstance(value, str):
        return redact_pii(value)[0]
    if isinstance(value, Mapping):
        return {key: anonymize_trace_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [anonymize_trace_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(anonymize_trace_payload(item) for item in value)
    return value


def configure_langsmith_tracing(enabled: bool) -> None:
    """Enable LangSmith with a PII anonymizer, or disable tracing entirely."""
    from langsmith import Client, configure

    if not enabled:
        configure(enabled=False)
        return

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise ValueError("LangSmith tracing requires LANGSMITH_API_KEY.")

    client = Client(anonymizer=anonymize_trace_payload)
    configure(client=client, enabled=True)
