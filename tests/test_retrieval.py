from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings
from retrieval import (
    BM25Index,
    IndexFingerprintMismatch,
    VectorStoreHandle,
    create_vector_store,
    index_chunks,
    load_indexed_chunks,
    remove_source,
)
from retrieval.indexing import chunk_id


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        base = [float(len(text) % 97), 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        return base


def make_settings(tmp_path: Path) -> Settings:
    from config import Settings

    return Settings(
        _env_file=None,
        project_root=tmp_path,
        vector_store_backend="chroma",
        top_k=2,
    )


def test_create_vector_store_uses_chroma(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())

    assert isinstance(handle, VectorStoreHandle)
    assert handle.backend_name == "chroma"
    assert handle.collection_name == "lawfirm_docs"


def test_index_chunks_is_incremental(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"

    chunks = [
        Document(page_content="billing rates", metadata={"chunk_id": "chunk-1"}),
        Document(page_content="filing fees", metadata={"chunk_id": "chunk-2"}),
    ]

    first = index_chunks(
        chunks,
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
    )
    second = index_chunks(
        chunks,
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
    )

    assert first.new_count == 2
    assert first.skipped_count == 0
    assert first.total_count == 2
    assert second.new_count == 0
    assert second.skipped_count == 2
    assert second.total_count == 2
    assert load_indexed_chunks(chunk_store_path).__len__() == 2


def test_index_chunks_force_reindexes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"

    chunks = [Document(page_content="billing rates", metadata={"chunk_id": "chunk-1"})]
    index_chunks(
        chunks, handle, bm25, manifest_path=manifest_path, chunk_store_path=chunk_store_path
    )

    forced = index_chunks(
        chunks,
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
        force=True,
    )

    assert forced.new_count == 1
    assert forced.total_count == 1


def test_index_chunks_replaces_stale_chunks_for_changed_document(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"

    old = Document(
        page_content="old billing policy",
        metadata={"chunk_id": "chunk-old", "source_file": "policy.md"},
    )
    new = Document(
        page_content="new billing policy",
        metadata={"chunk_id": "chunk-new", "source_file": "policy.md"},
    )

    first = index_chunks(
        [old], handle, bm25, manifest_path=manifest_path, chunk_store_path=chunk_store_path
    )
    second = index_chunks(
        [new], handle, bm25, manifest_path=manifest_path, chunk_store_path=chunk_store_path
    )

    stored = load_indexed_chunks(chunk_store_path)
    assert first.total_count == 1
    assert second.removed_count == 1
    assert second.new_count == 1
    assert second.total_count == 1
    assert [chunk.metadata["chunk_id"] for chunk in stored] == ["chunk-new"]
    assert bm25.document_count() == 1


def test_index_chunks_rejects_settings_fingerprint_mismatch(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"
    chunk = Document(
        page_content="billing rates",
        metadata={"chunk_id": "chunk-1", "source_file": "rates.md"},
    )

    index_chunks(
        [chunk],
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
        fingerprint="fingerprint-a",
    )

    with pytest.raises(IndexFingerprintMismatch):
        index_chunks(
            [chunk],
            handle,
            bm25,
            manifest_path=manifest_path,
            chunk_store_path=chunk_store_path,
            fingerprint="fingerprint-b",
        )

    rebuilt = index_chunks(
        [chunk],
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
        fingerprint="fingerprint-b",
        force=True,
    )
    assert rebuilt.fingerprint == "fingerprint-b"
    assert rebuilt.total_count == 1


def test_remove_source_deletes_only_that_documents_chunks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"
    chunks = [
        Document(
            page_content="billing rates",
            metadata={"chunk_id": "rates-1", "source_file": "rates.md"},
        ),
        Document(
            page_content="filing fees",
            metadata={"chunk_id": "rates-2", "source_file": "rates.md"},
        ),
        Document(
            page_content="office hours",
            metadata={"chunk_id": "office-1", "source_file": "office.md"},
        ),
    ]
    index_chunks(
        chunks, handle, bm25, manifest_path=manifest_path, chunk_store_path=chunk_store_path
    )

    removed = remove_source(
        "rates.md",
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
    )

    stored = load_indexed_chunks(chunk_store_path)
    assert removed == 2
    assert [chunk.metadata["source_file"] for chunk in stored] == ["office.md"]
    assert bm25.document_count() == 1
    remaining = handle.similarity_search("billing rates", k=5)
    assert all(document.metadata.get("source_file") != "rates.md" for document in remaining)


def test_remove_source_returns_zero_for_unknown_source(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    bm25 = BM25Index(k=2)
    manifest_path = tmp_path / "var" / "index" / "manifest.json"
    chunk_store_path = tmp_path / "var" / "index" / "chunks.jsonl"

    removed = remove_source(
        "missing.md",
        handle,
        bm25,
        manifest_path=manifest_path,
        chunk_store_path=chunk_store_path,
    )

    assert removed == 0


def test_bm25_retriever_returns_relevant_document() -> None:
    index = BM25Index(k=2)
    index.add_documents(
        [
            Document(page_content="billing rates for consultations", metadata={"id": "a"}),
            Document(page_content="filing fees for motions", metadata={"id": "b"}),
            Document(page_content="court deadlines and notices", metadata={"id": "c"}),
        ]
    )

    retriever = index.as_retriever()
    results = retriever.invoke("billing rates")

    assert results
    assert "billing" in results[0].page_content


def test_dense_retriever_returns_documents(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    handle = create_vector_store(settings, FakeEmbeddings())
    handle.add_documents(
        [Document(page_content="billing rates", metadata={"chunk_id": "chunk-1"})],
        ids=["chunk-1"],
    )

    retriever = handle.as_retriever(k=1)
    results = retriever.invoke("billing")

    assert len(results) == 1
    assert results[0].page_content == "billing rates"


def test_chunk_id_is_stable() -> None:
    document = Document(page_content="same content", metadata={})
    assert chunk_id(document) == chunk_id(Document(page_content="same content", metadata={}))
