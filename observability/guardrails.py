"""Input and output guardrails for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import get_settings


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: list[str]


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

        # Rate limiting is intentionally a placeholder hook for a future
        # production deployment. It never blocks in the local application.
        if not RateLimiterPlaceholder().allow():
            reasons.append("rate limit exceeded")

        return GuardrailResult(not reasons, reasons)


def detect_pii(text: str) -> list[str]:
    """Placeholder PII detector. Replace with a real detector in production."""

    return []


REFUSAL_MARKERS = (
    "do not have enough information",
    "cannot answer",
    "insufficient",
    "no information",
    "not have enough",
)


def is_refusal(answer: str) -> bool:
    """Return True when the answer is a refusal rather than a grounded answer."""

    normalized = (answer or "").lower()
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

        missing = [marker for marker in source_markers if marker not in answer]
        if len(missing) == len(source_markers):
            return GuardrailResult(False, ["answer does not cite any retrieved source"])
        return GuardrailResult(True, [])

    def validate_refusal(self, answer: str, documents: list[Any]) -> GuardrailResult:
        if documents:
            return GuardrailResult(True, [])
        if is_refusal(answer):
            return GuardrailResult(True, [])
        return GuardrailResult(False, ["answer was generated without supporting context"])
