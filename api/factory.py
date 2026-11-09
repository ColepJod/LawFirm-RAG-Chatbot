"""Build application state once at startup."""

from __future__ import annotations

import logging

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
from retrieval.cache import CachedReranker, CachedRetriever
from retrieval.hybrid import build_hybrid_retriever

logger = logging.getLogger(__name__)


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

    retriever = CachedRetriever(
        build_hybrid_retriever(settings, vector_store, bm25_index),
        max_entries=settings.retrieval_cache_size,
    )
    reranker = CachedReranker(
        create_reranker(settings),
        max_entries=settings.retrieval_cache_size,
    )
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

    state = AppState(
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
    warmup_state(state)
    return state


def warmup_state(state: AppState) -> None:
    """Load embedding/reranker models and prime the caches before the first query.

    This is best-effort: a missing model or offline machine must not prevent the
    API from starting. The first real request will surface the same error.
    """

    if not state.settings.warmup_on_startup:
        return
    query = state.settings.warmup_query
    try:
        documents = state.dependencies.retriever.invoke(query)
        reranker = state.dependencies.reranker
        if documents and reranker is not None and getattr(reranker, "enabled", False):
            reranker.compress_documents(documents, query)
        logger.info("model warmup complete query=%r documents=%d", query, len(documents))
    except Exception:
        logger.warning("model warmup failed; first request may be slower", exc_info=True)


def rebuild_retriever(state: AppState) -> None:
    """Refresh the retriever and graph after new documents are indexed."""

    with state.index_lock:
        new_retriever = CachedRetriever(
            build_hybrid_retriever(state.settings, state.vector_store, state.bm25_index),
            max_entries=state.settings.retrieval_cache_size,
        )
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
