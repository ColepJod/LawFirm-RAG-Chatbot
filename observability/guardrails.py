"""Input and output guardrails for the RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import get_settings


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: list[str]


CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[,\]])", re.IGNORECASE)
CITATION_PRESENT_PATTERN = re.compile(r"\[\s*\d+")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "phone": re.compile(r"\+?\d[\d .()-]{7,}\d"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "reveal your instructions",
    "show your system prompt",
    "print your system prompt",
    "you are now",
    "act as if you are",
    "jailbreak",
    "developer mode",
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return prompt-injection phrases found in user input."""

    normalized = (text or "").lower()
    return [pattern for pattern in INJECTION_PATTERNS if pattern in normalized]


class InputGuardrails:
    """Validate user input before it reaches the graph."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()

    def validate(self, query: str) -> GuardrailResult:
        reasons: list[str] = []

        if not isinstance(query, str) or not query.strip():
            reasons.append("query must be a non-empty string")
            return GuardrailResult(False, reasons)

        if len(query) > self.settings.max_query_chars:
            reasons.append(f"query exceeds {self.settings.max_query_chars} characters")

        if self.settings.pii_detection_enabled:
            pii_findings = detect_pii(query)
            if pii_findings:
                reasons.append(f"potential PII detected: {', '.join(pii_findings)}")

        injection_findings = detect_prompt_injection(query)
        if injection_findings:
            reasons.append("possible prompt injection: " + ", ".join(injection_findings[:3]))

        return GuardrailResult(not reasons, reasons)


def detect_pii(text: str) -> list[str]:
    """Return the names of PII categories found in ``text``."""

    findings: list[str] = []
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(text or ""):
            findings.append(name)
    return findings


REFUSAL_MARKERS = (
    "do not have enough information",
    "don't have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "insufficient",
    "no information",
    "no such word",
    "not mentioned in the indexed documents",
    "not in the indexed documents",
    "couldn't find",
    "could not find",
)


def is_refusal(answer: str) -> bool:
    """Return True when the answer is a pure refusal rather than a grounded answer.

    A mixed answer that refuses part of a question but then answers a related
    part with citations is not a pure refusal, so it should still be validated
    as a grounded answer.
    """

    normalized = (answer or "").lower()
    if CITATION_PRESENT_PATTERN.search(normalized):
        return False
    return any(marker in normalized for marker in REFUSAL_MARKERS)


class RateLimiterPlaceholder:
    """No-op rate limiter used until a real backend is configured."""

    def allow(self) -> bool:
        return True


class OutputGuardrails:
    """Validate generated answers before returning them."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()

    def validate_citations(self, answer: str, documents: list[Any]) -> GuardrailResult:
        if not documents:
            return GuardrailResult(True, [])

        source_markers = {
            str((getattr(document, "metadata", {}) or {}).get("source_file", ""))
            for document in documents
        }
        source_markers.discard("")
        if not source_markers:
            return GuardrailResult(True, [])

        # Refusals are legitimate grounded outcomes and do not need citations.
        if is_refusal(answer):
            return GuardrailResult(True, [])

        cited = any(marker in answer for marker in source_markers) or bool(
            CITATION_PATTERN.search(answer)
        )
        if not cited:
            return GuardrailResult(False, ["answer does not cite any retrieved source"])
        return GuardrailResult(True, [])

    def validate_refusal(
        self,
        answer: str,
        documents: list[Any],
        expect_context: bool | None = None,
    ) -> GuardrailResult:
        if documents:
            return GuardrailResult(True, [])
        if not expect_context:
            # Direct answers (chit-chat, out-of-scope) are not retrieval-based
            # and therefore do not require a refusal.
            return GuardrailResult(True, [])
        if is_refusal(answer):
            return GuardrailResult(True, [])
        return GuardrailResult(False, ["answer was generated without supporting context"])
