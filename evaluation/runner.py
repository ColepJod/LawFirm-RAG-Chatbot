"""Run the local-first evaluation harness."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from evaluation.golden_set import load_golden_set
from evaluation.llamella_scorer import score_response
from evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.thresholds import check_thresholds


@dataclass
class EvaluationOutput:
    """Result returned by an evaluated RAG pipeline."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    retrieved_source_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregate evaluation result."""

    passed: bool
    metrics: dict[str, float]
    item_results: list[dict[str, Any]]
    failures: list[str]
    output_path: Path


Pipeline = Callable[[str], EvaluationOutput]


def default_pipeline_factory(settings: Settings | None = None) -> Pipeline:
    """Build a pipeline from the configured graph and indexes."""

    settings = settings or get_settings()
    from graph import GraphDependencies, build_graph
    from retrieval import (
        BM25Index,
        create_embeddings,
        create_vector_store,
        load_indexed_chunks,
    )
    from retrieval.hybrid import build_hybrid_retriever

    embeddings = create_embeddings(settings)
    vector_store = create_vector_store(settings, embeddings)
    bm25 = BM25Index(k=settings.top_k)
    bm25.replace_documents(
        load_indexed_chunks(settings.project_root / "var" / "index" / "chunks.jsonl")
    )
    retriever = build_hybrid_retriever(settings, vector_store, bm25)

    from reranking import create_reranker

    reranker = create_reranker(settings)

    from generation import create_llm

    llm = create_llm(settings)

    deps = GraphDependencies(settings=settings, llm=llm, retriever=retriever, reranker=reranker)
    graph = build_graph(deps)

    def run(question: str) -> EvaluationOutput:
        result = graph.invoke(
            {"messages": [{"type": "human", "content": question}]},
            config={
                "configurable": {"thread_id": "evaluation"},
                "recursion_limit": settings.max_graph_steps,
            },
        )
        contexts = [
            chunk.page_content
            for chunk in result.get("reranked_chunks", []) or result.get("retrieved_chunks", [])
        ]
        return EvaluationOutput(
            answer=str(result.get("generation", "")),
            contexts=contexts,
            citations=list(result.get("citations", [])),
            retrieved_source_ids=list(result.get("retrieved_source_ids", [])),
        )

    return run


def run_evaluation(
    *,
    settings: Settings | None = None,
    pipeline: Pipeline | None = None,
    golden_set_path: Path | None = None,
    output_dir: Path | None = None,
    fail_on_regression: bool | None = None,
    k: int = 6,
) -> EvaluationReport:
    """Run evaluation and write a JSON report."""

    settings = settings or get_settings()
    golden_path = golden_set_path or settings.golden_set_path
    output = output_dir or settings.eval_output_dir
    output.mkdir(parents=True, exist_ok=True)
    should_fail = (
        settings.eval_fail_on_regression if fail_on_regression is None else fail_on_regression
    )

    items = load_golden_set(golden_path)
    pipeline = pipeline or default_pipeline_factory(settings)

    item_results: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "groundedness": [],
        "completeness": [],
        "relevance": [],
        "consistency": [],
        "confidence": [],
        "iqs": [],
        "retrieval_precision": [],
        "retrieval_recall": [],
        "reciprocal_rank": [],
        "hit_rate": [],
    }

    for item in items:
        result = pipeline(item.question)
        llamella_metrics = score_response(
            query=item.question,
            response=result.answer,
            contexts=result.contexts,
        )
        relevant = set(item.relevant_source_ids)
        retrieved = result.retrieved_source_ids or result.citations
        precision = precision_at_k(retrieved, relevant, k)
        recall = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)
        hit = hit_rate_at_k(retrieved, relevant, k)

        item_results.append(
            {
                "question": item.question,
                "category": item.category,
                "answer": result.answer,
                "citations": result.citations,
                "retrieved_source_ids": retrieved,
                "expected_answer": item.expected_answer,
                "relevant_source_ids": item.relevant_source_ids,
                "llamella": llamella_metrics,
                "retrieval": {
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "reciprocal_rank": rr,
                    "hit_rate_at_k": hit,
                },
            }
        )

        for metric in (
            "groundedness",
            "completeness",
            "relevance",
            "consistency",
            "confidence",
            "iqs",
        ):
            value = llamella_metrics.get(metric)
            if value is not None:
                metric_values[metric].append(float(value))
        metric_values["retrieval_precision"].append(precision)
        metric_values["retrieval_recall"].append(recall)
        metric_values["reciprocal_rank"].append(rr)
        metric_values["hit_rate"].append(hit)

    aggregate: dict[str, float] = {}
    for metric, values in metric_values.items():
        aggregate[metric] = (sum(values) / len(values)) if values else 0.0
    aggregate["retrieval_precision"] = aggregate["retrieval_precision"]

    failures = check_thresholds(aggregate, settings) if should_fail else []
    passed = not failures

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output / f"evaluation-{timestamp}.json"
    report = {
        "passed": passed,
        "metrics": aggregate,
        "thresholds": {
            "groundedness": settings.eval_groundedness_min,
            "completeness": settings.eval_completeness_min,
            "relevance": settings.eval_relevance_min,
            "consistency": settings.eval_consistency_min,
            "confidence": settings.eval_confidence_min,
            "iqs": settings.eval_iqs_min,
            "retrieval_precision": settings.eval_retrieval_precision_min,
        },
        "failures": failures,
        "item_count": len(items),
        "items": item_results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return EvaluationReport(
        passed=passed,
        metrics=aggregate,
        item_results=item_results,
        failures=failures,
        output_path=output_path,
    )
