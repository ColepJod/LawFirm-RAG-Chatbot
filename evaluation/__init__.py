"""Local-first evaluation harness (Step 9)."""

from evaluation.golden_set import GoldenItem, load_golden_set
from evaluation.llamella_scorer import score_response
from evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.runner import EvaluationOutput, EvaluationReport, run_evaluation
from evaluation.thresholds import check_thresholds

__all__ = [
    "EvaluationOutput",
    "EvaluationReport",
    "GoldenItem",
    "check_thresholds",
    "hit_rate_at_k",
    "load_golden_set",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_evaluation",
    "score_response",
]
