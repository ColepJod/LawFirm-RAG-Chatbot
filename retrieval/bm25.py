"""BM25 sparse retrieval over the same chunks used by dense retrieval.

This is a deliberately small wrapper around ``rank_bm25`` instead of the
deprecated ``langchain_community`` retriever. Keeping the index local gives us
deterministic tokenization, safe rebuilds, and score metadata that the hybrid
retriever can use without pulling in a sunset dependency.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*")

# A compact English stop-word set. Legal queries are dense with names, statute
# numbers, and defined terms, so pruning only the highest-frequency words keeps
# recall high while reducing noise.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for indexing and querying."""

    return [token for token in _TOKEN_RE.findall((text or "").lower()) if token not in STOPWORDS]


class BM25Retriever:
    """Minimal retriever interface used by the hybrid pipeline."""

    def __init__(self, index: BM25Index, k: int = 6) -> None:
        self.index = index
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        return self.index.search(query, k=self.k)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


class BM25Index:
    """An in-memory BM25 index with a LangChain-compatible retriever."""

    def __init__(self, k: int = 6, k1: float | None = None, b: float | None = None) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        from config import get_settings

        settings = get_settings()
        self.k = k
        self.k1 = settings.bm25_k1 if k1 is None else k1
        self.b = settings.bm25_b if b is None else b
        self.documents: list[Document] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)
        self._rebuild()

    def replace_documents(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        self._rebuild()

    def _rebuild(self) -> None:
        self._tokenized = [tokenize(document.page_content) for document in self.documents]
        if not self.documents or not any(self._tokenized):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Return the top ``k`` documents with a positive BM25 score."""

        if self._bm25 is None or not self.documents:
            return []

        limit = self.k if k is None else k
        if limit < 1:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            ((float(score), index) for index, score in enumerate(scores)),
            key=lambda item: item[0],
            reverse=True,
        )

        results: list[Document] = []
        for score, index in ranked:
            if score <= 0.0:
                break
            document = self.documents[index]
            metadata = dict(document.metadata or {})
            metadata["bm25_score"] = score
            results.append(Document(page_content=document.page_content, metadata=metadata))
            if len(results) >= limit:
                break
        return results

    def as_retriever(self, k: int | None = None) -> BM25Retriever:
        if self._bm25 is None:
            raise ValueError("BM25 index is empty")
        return BM25Retriever(self, k=self.k if k is None else k)

    def document_count(self) -> int:
        return len(self.documents)

    def configure(
        self,
        *,
        k: int | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        """Update BM25 parameters and rebuild the index."""

        if k is not None:
            if k < 1:
                raise ValueError("k must be positive")
            self.k = k
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b
        self._rebuild()

    def status(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count(),
            "k": self.k,
            "k1": self.k1,
            "b": self.b,
            "ready": self._bm25 is not None,
        }
