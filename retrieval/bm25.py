"""BM25 sparse retrieval over the same chunks used by dense retrieval."""

from __future__ import annotations

from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document


class BM25Index:
    """An in-memory BM25 index with a LangChain retriever interface."""

    def __init__(self, k: int = 6) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.documents: list[Document] = []
        self._retriever: BM25Retriever | None = None

    def add_documents(self, documents: list[Document]) -> None:
        self.documents.extend(documents)
        self._rebuild()

    def replace_documents(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.documents:
            self._retriever = None
            return
        self._retriever = BM25Retriever.from_documents(self.documents, k=self.k)

    def as_retriever(self):
        if self._retriever is None:
            raise ValueError("BM25 index is empty")
        return self._retriever

    def document_count(self) -> int:
        return len(self.documents)
