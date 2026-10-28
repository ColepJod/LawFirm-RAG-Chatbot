"""Build application state once at startup."""

from __future__ import annotations

from api.state import AppState
from config import apply_model_cache_environment, get_settings
from graph import GraphDependencies, build_graph
from observability import InputGuardrails, OutputGuardrails, TraceLogger
from reranking import create_reranker
from retrieval import (
    BM25Index,
    create_embeddings,
    create_vector_store,
    load_indexed_chunks,
)
from retrieval.hybrid import build_hybrid_retriever


def build_app_state(settings=None) -> AppState:
    settings = settings or get_settings()
    apply_model_cache_environment(settings)

    embeddings = create_embeddings(settings)
    vector_store = create_vector_store(settings, embeddings)
    bm25_index = BM25Index(k=settings.top_k)
    chunk_store = settings.project_root / "var" / "index" / "chunks.jsonl"
    bm25_index.replace_documents(load_indexed_chunks(chunk_store))

    retriever = build_hybrid_retriever(settings, vector_store, bm25_index)
    reranker = create_reranker(settings)
    trace_logger = TraceLogger(settings)

    dependencies = GraphDependencies(
        settings=settings,
        llm=_create_llm(settings),
        retriever=retriever,
        reranker=reranker,
        trace_logger=trace_logger,
    )
    graph = build_graph(dependencies)

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
    )


def _create_llm(settings):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )


def rebuild_retriever(state: AppState) -> None:
    """Refresh the retriever after new documents are indexed."""

    new_retriever = build_hybrid_retriever(state.settings, state.vector_store, state.bm25_index)
    state.dependencies.retriever = new_retriever
    state.graph = build_graph(state.dependencies)
