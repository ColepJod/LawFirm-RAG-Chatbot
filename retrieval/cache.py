"""Small thread-safe LRU caches for retrieval and reranking.

Repeated questions are common in a firm chat (opening hours, rates, contact
details), and both hybrid retrieval and cross-encoder reranking are
deterministic for a fixed index. Caching their results removes the two largest
per-query costs after the model warmup.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

from langchain_core.documents import Document


def normalize_query(query: str) -> str:
    """Normalize whitespace so cosmetic formatting does not split cache entries."""

    return " ".join((query or "").split())


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("chunk_id", "id", "document_hash"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value}"
    digest = hashlib.sha256((document.page_content or "").encode("utf-8")).hexdigest()
    return f"content:{digest}"


def copy_documents(documents: list[Document]) -> list[Document]:
    """Copy documents so a caller cannot mutate cached metadata."""

    return [
        Document(page_content=document.page_content, metadata=dict(document.metadata or {}))
        for document in documents
    ]


class LRUCache:
    """A tiny LRU cache guarded by a lock."""

    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(0, max_entries)
        self._data: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def get(self, key: Any) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._data),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
            }


class CachedRetriever:
    """Cache hybrid retrieval results by normalized query."""

    def __init__(self, inner: Any, max_entries: int = 128) -> None:
        self.inner = inner
        self.cache = LRUCache(max_entries)

    def invoke(self, query: str) -> list[Document]:
        key = normalize_query(query)
        cached = self.cache.get(key)
        if cached is not None:
            return copy_documents(cached)
        documents = list(self.inner.invoke(query))
        self.cache.set(key, copy_documents(documents))
        return documents

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)

    def stats(self) -> dict[str, int]:
        return self.cache.stats()


class CachedReranker:
    """Cache cross-encoder results for a query/document-set pair."""

    def __init__(self, inner: Any, max_entries: int = 128) -> None:
        self.inner = inner
        self.cache = LRUCache(max_entries)
        self.enabled = bool(getattr(inner, "enabled", True))

    def compress_documents(self, documents, query, callbacks=None):
        docs = list(documents or [])
        if not docs:
            return []
        key = (normalize_query(query), tuple(document_key(document) for document in docs))
        cached = self.cache.get(key)
        if cached is not None:
            return copy_documents(cached)
        result = list(self.inner.compress_documents(docs, query))
        self.cache.set(key, copy_documents(result))
        return result

    def stats(self) -> dict[str, int]:
        return self.cache.stats()
