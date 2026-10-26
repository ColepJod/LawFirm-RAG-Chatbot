"""Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("chunk_id", "id", "document_hash"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value}"
    return f"content:{hash(document.page_content)}"


def rrf_fuse(
    dense_documents: list[Document],
    sparse_documents: list[Document],
    *,
    k: int = 60,
    top_k: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
) -> list[Document]:
    """Fuse two ranked document lists using Reciprocal Rank Fusion."""

    if k < 1:
        raise ValueError("RRF k must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(weights) != 2 or any(weight < 0 for weight in weights):
        raise ValueError("weights must contain two non-negative values")
    if sum(weights) <= 0:
        raise ValueError("at least one RRF weight must be greater than zero")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for rank, document in enumerate(dense_documents, start=1):
        key = _document_key(document)
        scores[key] = scores.get(key, 0.0) + (weights[0] / (k + rank))
        documents[key] = document

    for rank, document in enumerate(sparse_documents, start=1):
        key = _document_key(document)
        scores[key] = scores.get(key, 0.0) + (weights[1] / (k + rank))
        if key not in documents:
            documents[key] = document

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

    fused: list[Document] = []
    for key in ordered_keys[:top_k]:
        original = documents[key]
        metadata = dict(original.metadata or {})
        metadata["rrf_score"] = scores[key]
        fused.append(Document(page_content=original.page_content, metadata=metadata))

    return fused


class HybridRRFRetriever(BaseRetriever):
    """A LangChain retriever that fuses dense and BM25 results with RRF."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_retriever: Any
    sparse_retriever: Any
    top_k: int = 6
    rrf_k: int = 60
    weights: tuple[float, float] = (0.5, 0.5)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        dense_documents = self.dense_retriever.invoke(query)
        sparse_documents = self.sparse_retriever.invoke(query)
        return rrf_fuse(
            dense_documents,
            sparse_documents,
            k=self.rrf_k,
            top_k=self.top_k,
            weights=self.weights,
        )
