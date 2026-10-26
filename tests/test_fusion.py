from __future__ import annotations

from langchain_core.documents import Document

from retrieval import HybridRRFRetriever, rrf_fuse


def test_rrf_fuse_combines_ranked_lists() -> None:
    dense = [
        Document(page_content="billing rates", metadata={"chunk_id": "a"}),
        Document(page_content="filing fees", metadata={"chunk_id": "b"}),
    ]
    sparse = [
        Document(page_content="billing rates", metadata={"chunk_id": "a"}),
        Document(page_content="court deadlines", metadata={"chunk_id": "c"}),
    ]

    fused = rrf_fuse(dense, sparse, k=60, top_k=3, weights=(0.5, 0.5))

    assert [doc.metadata["chunk_id"] for doc in fused] == ["a", "b", "c"]
    assert all("rrf_score" in doc.metadata for doc in fused)
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_rrf_fuse_deduplicates_documents() -> None:
    dense = [Document(page_content="same", metadata={"chunk_id": "x"})]
    sparse = [Document(page_content="same", metadata={"chunk_id": "x"})]

    fused = rrf_fuse(dense, sparse, top_k=5)

    assert len(fused) == 1


def test_rrf_fuse_rejects_invalid_weights() -> None:
    import pytest

    with pytest.raises(ValueError):
        rrf_fuse([], [], weights=(0.0, 0.0))


def test_rrf_fuse_top_k_limits_results() -> None:
    dense = [Document(page_content=f"doc {i}", metadata={"chunk_id": f"d{i}"}) for i in range(5)]
    fused = rrf_fuse(dense, [], top_k=2)

    assert len(fused) == 2


def test_hybrid_retriever_invokes_both_sources() -> None:
    class FakeRetriever:
        def __init__(self, docs):
            self.docs = docs

        def invoke(self, query):
            return self.docs

    dense_docs = [Document(page_content="dense", metadata={"chunk_id": "d"})]
    sparse_docs = [Document(page_content="sparse", metadata={"chunk_id": "s"})]
    retriever = HybridRRFRetriever(
        dense_retriever=FakeRetriever(dense_docs),
        sparse_retriever=FakeRetriever(sparse_docs),
        top_k=2,
        rrf_k=60,
    )

    result = retriever.invoke("query")

    assert {doc.metadata["chunk_id"] for doc in result} == {"d", "s"}
