"""Threshold checks for evaluation reports."""

from __future__ import annotations

from config import Settings


def check_thresholds(metrics: dict[str, float], settings: Settings) -> list[str]:
    """Return a list of failed threshold descriptions."""

    failures: list[str] = []
    checks = {
        "groundedness": settings.eval_groundedness_min,
        "completeness": settings.eval_completeness_min,
        "relevance": settings.eval_relevance_min,
        "consistency": settings.eval_consistency_min,
        "confidence": settings.eval_confidence_min,
        "iqs": settings.eval_iqs_min,
        "retrieval_precision": settings.eval_retrieval_precision_min,
    }
    for metric, threshold in checks.items():
        value = metrics.get(metric)
        if value is not None and value < threshold:
            failures.append(f"{metric}={value:.3f} < {threshold:.3f}")
    return failures
