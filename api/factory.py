"""Build application state once at startup."""

from __future__ import annotations

from api.security import TokenBucketRateLimiter
from api.state import AppState
from config import apply_model_cache_environment, get_settings
from generation import create_llm
from graph import GraphDependencies, build_graph, create_checkpointer
from observability import InputGuardrails, OutputGuardrails, TraceLogger
from reranking import create_reranker
from retrieval import (
    BM25Index,
    IndexManifest,
    create_embeddings,
    create_vector_store,
    load_indexed_chunks,
)
from retrieval.hybrid import build_hybrid_retriever


def build_app_state(settings=None) -> AppState:
    settings = settings or get_settings()
    apply_model_cache_environment(settings)
    settings.ensure_directories()

    embeddings = create_embeddings(settings)
    vector_store = create_vector_store(settings, embeddings)
    bm25_index = BM25Index(k=settings.top_k, k1=settings.bm25_k1, b=settings.bm25_b)
    chunk_store = settings.index_dir / "chunks.jsonl"
    manifest_path = settings.index_dir / "manifest.json"
    bm25_index.replace_documents(load_indexed_chunks(chunk_store))

    manifest = IndexManifest.load(manifest_path)
    fingerprint = settings.index_fingerprint
    fingerprint_ok = manifest.fingerprint is None or manifest.fingerprint == fingerprint

    retriever = build_hybrid_retriever(settings, vector_store, bm25_index)
    reranker = create_reranker(settings)
    trace_logger = TraceLogger(settings)
    checkpointer = create_checkpointer(settings)

    dependencies = GraphDependencies(
        settings=settings,
        llm=create_llm(settings),
        retriever=retriever,
        reranker=reranker,
        trace_logger=trace_logger,
    )
    graph = build_graph(dependencies, checkpointer=checkpointer)

    return AppState(
        settings=settings,
        graph=graph,
        dependencies=dependencies,
        vector_store=vector_store,
        bm25_index=bm25_index,
        trace_logger=trace_logger,
        input_guardrails=InputGuardrails(settings),
        output_guardrails=OutputGuardrails(settings),
        trace_path=settings.trace_dir / "traces.jsonl",
        checkpointer=checkpointer,
        index_manifest=manifest,
        index_fingerprint=fingerprint,
        index_fingerprint_ok=fingerprint_ok,
        rate_limiter=TokenBucketRateLimiter(settings.rate_limit_per_minute),
    )


def rebuild_retriever(state: AppState) -> None:
    """Refresh the retriever and graph after new documents are indexed."""

    with state.index_lock:
        new_retriever = build_hybrid_retriever(state.settings, state.vector_store, state.bm25_index)
        state.dependencies.retriever = new_retriever
        # Reuse the existing checkpointer; otherwise every reindex silently
        # discards the conversation state.
        state.graph = build_graph(state.dependencies, checkpointer=state.checkpointer)
        state.index_manifest = IndexManifest.load(state.settings.index_dir / "manifest.json")
        state.index_fingerprint = state.settings.index_fingerprint
        state.index_fingerprint_ok = (
            state.index_manifest.fingerprint is None
            or state.index_manifest.fingerprint == state.index_fingerprint
        )
