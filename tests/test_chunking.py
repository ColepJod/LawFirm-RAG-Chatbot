from __future__ import annotations

from langchain_core.documents import Document

from chunking import chunk_documents, count_tokens
from chunking.recursive import RecursiveTokenChunker
from chunking.structure import StructureAwareChunker


def _doc(text: str, **metadata: object) -> Document:
    base = {"source_file": "guide.md", "document_hash": "abc123", "section_heading": None}
    base.update(metadata)
    return Document(page_content=text, metadata=base)


def test_recursive_chunker_never_exceeds_token_budget() -> None:
    text = " ".join(f"Sentence number {i} explains a billing rule." for i in range(200))
    document = _doc(text)

    chunks = RecursiveTokenChunker(chunk_size=64, chunk_overlap=8).split_documents([document])

    assert chunks
    assert all(chunk.metadata["chunk_token_count"] <= 64 for chunk in chunks)
    assert all("Document: guide.md | Section:" in chunk.page_content for chunk in chunks)


def test_recursive_chunker_preserves_table_rows() -> None:
    text = (
        "Billing table:\n\n"
        "| Service | Rate |\n"
        "| --- | --- |\n"
        "| Consultation | $300 |\n"
        "| Filing | $150 |\n"
        "| Review | $200 |\n"
    )
    document = _doc(text)

    chunks = RecursiveTokenChunker(chunk_size=40, chunk_overlap=4).split_documents([document])

    rows = ["| Consultation | $300 |", "| Filing | $150 |", "| Review | $200 |"]
    joined = "\n".join(chunk.page_content for chunk in chunks)
    for row in rows:
        assert row in joined
    assert all(
        "| Consultation | $300 |" not in chunk.page_content or chunk.page_content.count("|") >= 2
        for chunk in chunks
    )


def test_structure_chunker_splits_by_headings() -> None:
    text = (
        "# Billing\n\n"
        "Rates are listed below.\n\n"
        "## Consultations\n\n"
        "Consultations cost $300.\n\n"
        "## Filings\n\n"
        "Filings cost $150.\n"
    )
    document = _doc(text, section_heading=None)

    chunks = StructureAwareChunker(chunk_size=512, chunk_overlap=16).split_documents([document])

    headings = {chunk.metadata.get("section_heading") for chunk in chunks}
    assert "Billing" in headings or "Consultations" in headings
    assert len(chunks) >= 2
    assert all(chunk.metadata["chunk_strategy"] == "structure" for chunk in chunks)


def test_structure_chunker_falls_back_for_plain_text() -> None:
    text = " ".join(f"Plain paragraph {i}." for i in range(100))
    document = _doc(text)

    chunks = StructureAwareChunker(chunk_size=48, chunk_overlap=8).split_documents([document])

    assert chunks
    assert all(chunk.metadata["chunk_token_count"] <= 48 for chunk in chunks)


def test_chunk_metadata_contains_required_fields() -> None:
    document = _doc("A short document about billing.")

    chunks = chunk_documents([document], strategy="recursive", chunk_size=128, chunk_overlap=16)

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["chunk_strategy"] == "recursive"
    assert metadata["chunk_index"] == 0
    assert metadata["chunk_token_count"] == count_tokens(chunks[0].page_content)
    assert metadata["chunk_id"]
    assert metadata["section_heading"] == "document"


def test_chunk_documents_uses_configured_defaults(monkeypatch: object) -> None:
    document = _doc("Default configuration chunking.")

    chunks = chunk_documents([document])

    assert chunks
    assert chunks[0].metadata["chunk_strategy"] in {"recursive", "structure"}


def test_empty_document_does_not_crash() -> None:
    document = _doc("")

    chunks = chunk_documents([document], strategy="recursive", chunk_size=64, chunk_overlap=8)

    assert len(chunks) == 1
    assert "Document: guide.md | Section:" in chunks[0].page_content
