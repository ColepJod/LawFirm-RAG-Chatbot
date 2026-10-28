from __future__ import annotations

import json
from pathlib import Path

from config import Settings
from evaluation import (
    check_thresholds,
    load_golden_set,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
)
from evaluation.runner import EvaluationOutput


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, project_root=tmp_path, eval_output_dir=tmp_path / "eval")


def test_precision_recall_and_reciprocal_rank() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_load_golden_set(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_answer": "A1", "relevant_source_ids": ["s1"], '
        '"category": "billing"}\n',
        encoding="utf-8",
    )

    items = load_golden_set(path)

    assert len(items) == 1
    assert items[0].question == "Q1"
    assert items[0].relevant_source_ids == ["s1"]


def test_run_evaluation_with_fake_pipeline(tmp_path: Path, monkeypatch) -> None:
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(
        '{"question": "Q1", "expected_answer": "A1", "relevant_source_ids": ["s1"], '
        '"category": "billing"}\n',
        encoding="utf-8",
    )
    settings = make_settings(tmp_path)

    def fake_score(query, response, contexts, **kwargs):
        return {
            "groundedness": 0.9,
            "completeness": 0.9,
            "relevance": 0.9,
            "consistency": 0.9,
            "confidence": 0.9,
            "iqs": 0.9,
        }

    monkeypatch.setattr("evaluation.runner.score_response", fake_score)

    def pipeline(question: str) -> EvaluationOutput:
        return EvaluationOutput(
            answer="A1",
            contexts=["context"],
            citations=["s1"],
            retrieved_source_ids=["s1"],
        )

    report = run_evaluation(
        settings=settings,
        pipeline=pipeline,
        golden_set_path=golden_path,
        output_dir=settings.eval_output_dir,
        fail_on_regression=True,
        k=2,
    )

    assert report.passed is True
    assert report.metrics["retrieval_precision"] == 1.0
    assert report.output_path.exists()
    saved = json.loads(report.output_path.read_text(encoding="utf-8"))
    assert saved["passed"] is True


def test_check_thresholds_reports_failures(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    metrics = {
        "groundedness": 0.5,
        "completeness": 0.9,
        "relevance": 0.9,
        "consistency": 0.9,
        "confidence": 0.9,
        "iqs": 0.9,
        "retrieval_precision": 0.9,
    }

    failures = check_thresholds(metrics, settings)

    assert any("groundedness" in failure for failure in failures)
