from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage

from chunking import chunk_documents
from config import Settings
from graph import GraphDependencies, build_graph
from ingestion import ingest_directory
from retrieval import (
    BM25Index,
    create_vector_store,
    index_chunks,
)
from retrieval.hybrid import build_hybrid_retriever
from tests.test_retrieval import FakeEmbeddings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None, project_root=tmp_path, vector_store_backend="chroma", top_k=2, **overrides
    )


def test_end_to_end_ingestion_chunking_indexing_and_retrieval(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "billing.md").write_text(
        "# Billing\n\nConsultations cost $300 per hour.\n\nFiling a motion costs $150.",
        encoding="utf-8",
    )
    (corpus / "locations.md").write_text(
        "# Locations\n\nThe Chicago office serves Midwest clients.",
        encoding="utf-8",
    )

    settings = make_settings(tmp_path)
    ingestion_result = ingest_directory(corpus, manifest_path=tmp_path / "manifest.json")
    assert len(ingestion_result.documents) == 2

    chunks = chunk_documents(
        ingestion_result.documents, strategy="structure", chunk_size=128, chunk_overlap=16
    )
    assert chunks
    assert all(chunk.metadata["chunk_token_count"] <= 128 for chunk in chunks)

    vector_store = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=settings.top_k)
    result = index_chunks(
        chunks,
        vector_store,
        bm25,
        manifest_path=tmp_path / "index_manifest.json",
        chunk_store_path=tmp_path / "chunks.jsonl",
    )
    assert result.new_count == len(chunks)
    assert result.total_count == len(chunks)

    retriever = build_hybrid_retriever(settings, vector_store, bm25)
    retrieved = retriever.invoke("consultation rate")
    source_files = {doc.metadata.get("source_file") for doc in retrieved}

    assert "billing.md" in source_files
    assert len(retrieved) <= settings.top_k


def test_end_to_end_graph_refuses_without_context() -> None:
    import tempfile
    from pathlib import Path

    settings = make_settings(Path(tempfile.mkdtemp()), max_query_rewrites=0)

    class FakeLLM:
        def __init__(self):
            self.responses = ["rag"]

        @property
        def _llm_type(self):
            return "fake"

        def invoke(self, messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content=self.responses.pop(0))

    retriever = type("R", (), {"invoke": lambda self, q: []})()
    reranker = type("Rr", (), {"compress_documents": lambda self, docs, q: list(docs)})()

    deps = GraphDependencies(
        settings=settings,
        llm=FakeLLM(),
        retriever=retriever,
        reranker=reranker,
    )
    graph = build_graph(deps)
    result = graph.invoke(
        {"messages": [HumanMessage("What is the secret rate?")]},
        config={"configurable": {"thread_id": "e2e"}, "recursion_limit": 20},
    )

    assert "do not have enough information" in result["generation"].lower()
    assert result["grounding_verdict"] == "pass"


def test_end_to_end_retrieval_returns_expected_source_in_top_results(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "billing.md").write_text("Consultations cost $300 per hour.", encoding="utf-8")
    (corpus / "locations.md").write_text(
        "The Chicago office serves Midwest clients.", encoding="utf-8"
    )

    settings = make_settings(tmp_path)
    ingestion_result = ingest_directory(corpus, manifest_path=tmp_path / "manifest.json")
    chunks = chunk_documents(
        ingestion_result.documents, strategy="recursive", chunk_size=64, chunk_overlap=8
    )
    vector_store = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    index_chunks(
        chunks,
        vector_store,
        bm25,
        manifest_path=tmp_path / "index_manifest.json",
        chunk_store_path=tmp_path / "chunks.jsonl",
    )

    retriever = build_hybrid_retriever(settings, vector_store, bm25)
    results = retriever.invoke("consultation")

    assert any(doc.metadata.get("source_file") == "billing.md" for doc in results[:3])
