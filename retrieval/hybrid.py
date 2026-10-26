"""Hybrid dense + BM25 retrieval fused with RRF."""

from __future__ import annotations

from config import get_settings
from retrieval.bm25 import BM25Index
from retrieval.fusion import HybridRRFRetriever
from retrieval.vectorstores import VectorStoreHandle


def build_hybrid_retriever(
    settings=None,
    vector_store: VectorStoreHandle | None = None,
    bm25_index: BM25Index | None = None,
):
    """Build an RRF-fused retriever from settings and indexes."""

    settings = settings or get_settings()
    if vector_store is None or bm25_index is None:
        raise ValueError("vector_store and bm25_index are required")

    dense_retriever = vector_store.as_retriever(k=settings.dense_candidates)
    sparse_retriever = bm25_index.as_retriever() if bm25_index.document_count() else None
    if sparse_retriever is None:
        return dense_retriever

    bm25_index.k = settings.sparse_candidates
    bm25_index._rebuild()
    sparse_retriever = bm25_index.as_retriever()

    return HybridRRFRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=settings.top_k,
        rrf_k=settings.rrf_k,
        weights=(settings.dense_weight, settings.sparse_weight),
    )
