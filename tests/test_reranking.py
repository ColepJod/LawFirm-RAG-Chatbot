from __future__ import annotations

from langchain_core.documents import Document

from reranking import CrossEncoderReranker, create_reranker, rerank_documents


class FakeCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        return [self.scores[text] for _, text in pairs]


def test_reranker_orders_by_cross_encoder_score() -> None:
    docs = [
        Document(page_content="low relevance", metadata={"chunk_id": "a"}),
        Document(page_content="high relevance", metadata={"chunk_id": "b"}),
    ]
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder({"low relevance": 0.1, "high relevance": 0.9}),
        top_n=2,
    )

    result = reranker.compress_documents(docs, "query")

    assert [doc.metadata["chunk_id"] for doc in result] == ["b", "a"]
    assert result[0].metadata["reranker_score"] == 0.9


def test_reranker_respects_top_n() -> None:
    docs = [Document(page_content=f"doc {i}", metadata={"chunk_id": f"d{i}"}) for i in range(5)]
    scores = {f"doc {i}": float(i) for i in range(5)}
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(scores), top_n=3)

    result = reranker.compress_documents(docs, "query")

    assert len(result) == 3
    assert [doc.metadata["chunk_id"] for doc in result] == ["d4", "d3", "d2"]


def test_disabled_reranker_returns_original_prefix() -> None:
    docs = [Document(page_content=f"doc {i}", metadata={"chunk_id": f"d{i}"}) for i in range(5)]
    reranker = CrossEncoderReranker(enabled=False, top_n=2)

    result = reranker.compress_documents(docs, "query")

    assert [doc.metadata["chunk_id"] for doc in result] == ["d0", "d1"]


def test_empty_documents_return_empty() -> None:
    reranker = CrossEncoderReranker(top_n=3)

    assert reranker.compress_documents([], "query") == []


def test_rerank_documents_helper_uses_provided_reranker() -> None:
    docs = [Document(page_content="only", metadata={"chunk_id": "x"})]
    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder({"only": 0.42}),
        top_n=1,
    )

    result = rerank_documents(docs, "query", reranker=reranker)

    assert result[0].metadata["reranker_score"] == 0.42


def test_create_reranker_reads_settings() -> None:
    from config import Settings

    settings = Settings(_env_file=None, top_k=2, reranker_model="custom/model", rerank_top_n=4)
    reranker = create_reranker(settings)

    assert reranker.model_name == "custom/model"
    assert reranker.top_n == 4
