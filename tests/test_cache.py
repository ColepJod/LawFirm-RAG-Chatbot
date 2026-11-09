from __future__ import annotations

from langchain_core.documents import Document

from retrieval.cache import CachedReranker, CachedRetriever, LRUCache


class CountingRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls = 0

    def invoke(self, query: str) -> list[Document]:
        self.calls += 1
        return [
            Document(page_content=document.page_content, metadata=dict(document.metadata))
            for document in self.documents
        ]


class CountingReranker:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    def compress_documents(self, documents, query):
        self.calls += 1
        return list(documents)


def test_lru_cache_evicts_oldest_entry() -> None:
    cache = LRUCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.stats()["entries"] == 2


def test_lru_cache_zero_disables_caching() -> None:
    cache = LRUCache(max_entries=0)
    cache.set("a", 1)

    assert cache.enabled is False
    assert cache.get("a") is None


def test_cached_retriever_reuses_results_and_returns_copies() -> None:
    inner = CountingRetriever([Document(page_content="billing rates", metadata={"chunk_id": "a"})])
    retriever = CachedRetriever(inner, max_entries=8)

    first = retriever.invoke("billing rates")
    first[0].metadata["mutated"] = True
    second = retriever.invoke("billing   rates")

    assert inner.calls == 1
    assert "mutated" not in second[0].metadata
    assert retriever.stats()["hits"] == 1


def test_cached_reranker_reuses_results_for_identical_documents() -> None:
    inner = CountingReranker()
    reranker = CachedReranker(inner, max_entries=8)
    documents = [Document(page_content="billing", metadata={"chunk_id": "a"})]

    first = reranker.compress_documents(documents, "rates")
    second = reranker.compress_documents(documents, "rates")

    assert inner.calls == 1
    assert first[0].page_content == second[0].page_content
    assert reranker.stats()["hits"] == 1


def test_cached_reranker_distinguishes_document_sets() -> None:
    inner = CountingReranker()
    reranker = CachedReranker(inner, max_entries=8)
    first_docs = [Document(page_content="billing", metadata={"chunk_id": "a"})]
    second_docs = [Document(page_content="filing", metadata={"chunk_id": "b"})]

    reranker.compress_documents(first_docs, "rates")
    reranker.compress_documents(second_docs, "rates")

    assert inner.calls == 2
