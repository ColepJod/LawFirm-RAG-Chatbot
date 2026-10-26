from __future__ import annotations

from pathlib import Path

import fitz
from langchain_core.documents import Document

from ingestion import (
    IngestionResult,
    discover_documents,
    ingest_directory,
    ingest_file,
)


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_discover_documents_filters_supported_extensions(tmp_path: Path) -> None:
    _write_text(tmp_path / "a.txt", "text")
    _write_text(tmp_path / "b.md", "# Title")
    _write_text(tmp_path / "ignore.csv", "x,y")
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_text(nested / "c.csv", "not supported")

    found = {path.name for path in discover_documents(tmp_path)}

    assert found == {"a.txt", "b.md"}


def test_ingest_text_file_attaches_required_metadata(tmp_path: Path) -> None:
    source = _write_text(tmp_path / "note.txt", "A simple text document.")
    manifest = tmp_path / "manifest.json"

    documents = ingest_file(source, manifest_path=manifest)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert document.page_content == "A simple text document."
    assert document.metadata["source_file"] == "note.txt"
    assert document.metadata["file_type"] == "txt"
    assert document.metadata["page_number"] is None
    assert document.metadata["section_heading"] is None
    assert document.metadata["document_hash"]
    assert document.metadata["ingestion_timestamp"]
    assert manifest.exists()


def test_ingest_markdown_extracts_first_heading(tmp_path: Path) -> None:
    source = _write_text(tmp_path / "guide.md", "# Billing Guide\n\nRates are listed here.")
    manifest = tmp_path / "manifest.json"

    documents = ingest_file(source, manifest_path=manifest)

    assert documents[0].metadata["section_heading"] == "Billing Guide"
    assert documents[0].metadata["file_type"] == "md"


def test_ingest_is_idempotent_by_hash(tmp_path: Path) -> None:
    source = _write_text(tmp_path / "note.txt", "same content")
    manifest = tmp_path / "manifest.json"

    first = ingest_file(source, manifest_path=manifest)
    second = ingest_file(source, manifest_path=manifest)

    assert len(first) == 1
    assert second == []


def test_force_reingests_existing_document(tmp_path: Path) -> None:
    source = _write_text(tmp_path / "note.txt", "same content")
    manifest = tmp_path / "manifest.json"

    ingest_file(source, manifest_path=manifest)
    forced = ingest_file(source, manifest_path=manifest, force=True)

    assert len(forced) == 1


def test_ingest_directory_returns_only_new_documents(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_text(corpus / "a.txt", "first")
    _write_text(corpus / "b.txt", "second")
    manifest = tmp_path / "manifest.json"

    first = ingest_directory(corpus, manifest_path=manifest)
    second = ingest_directory(corpus, manifest_path=manifest)

    assert isinstance(first, IngestionResult)
    assert len(first.documents) == 2
    assert first.skipped == []
    assert second.documents == []
    assert len(second.skipped) == 2


def test_ingest_deduplicates_identical_files_in_one_directory(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_text(corpus / "a.txt", "duplicate")
    _write_text(corpus / "b.txt", "duplicate")
    manifest = tmp_path / "manifest.json"

    result = ingest_directory(corpus, manifest_path=manifest)

    assert len(result.documents) == 1
    assert len(result.skipped) == 1


def test_pdf_ingestion_extracts_text_and_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Section 1: Billing rates")
    document.save(pdf_path)
    document.close()

    manifest = tmp_path / "manifest.json"
    documents = ingest_file(pdf_path, manifest_path=manifest, pdf_parser="pdfmux")

    assert documents
    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].metadata["page_number"] == 1
    assert "Billing rates" in documents[0].page_content
    assert documents[0].metadata["section_heading"]


def test_manifest_is_json_and_records_document_count(tmp_path: Path) -> None:
    source = _write_text(tmp_path / "note.txt", "content")
    manifest = tmp_path / "manifest.json"

    ingest_file(source, manifest_path=manifest)

    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data) == 1
    entry = next(iter(data.values()))
    assert entry["document_count"] == 1
    assert entry["file_type"] == "txt"
