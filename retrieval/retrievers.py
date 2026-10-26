"""LangChain retriever construction for dense and sparse retrieval."""

from __future__ import annotations

from config import get_settings
from retrieval.bm25 import BM25Index
from retrieval.vectorstores import VectorStoreHandle


def create_dense_retriever(vector_store: VectorStoreHandle, k: int | None = None):
    settings = get_settings()
    top_k = settings.top_k if k is None else k
    return vector_store.as_retriever(k=top_k)


def create_bm25_retriever(bm25_index: BM25Index, k: int | None = None):
    settings = get_settings()
    top_k = settings.top_k if k is None else k
    bm25_index.k = top_k
    bm25_index._rebuild()
    return bm25_index.as_retriever()


def build_retrievers(settings, vector_store: VectorStoreHandle, bm25_index: BM25Index):
    return {
        "dense": create_dense_retriever(vector_store, settings.top_k),
        "bm25": create_bm25_retriever(bm25_index, settings.top_k),
    }
