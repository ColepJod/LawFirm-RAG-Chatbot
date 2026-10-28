"""Retrieval and response metrics for local evaluation."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / min(k, len(retrieved))


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 1.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
