"""Idempotent document ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from config import get_settings
from ingestion.discovery import discover_documents
from ingestion.hashing import sha256_file
from ingestion.manifest import IngestionManifest
from ingestion.metadata import base_metadata, relative_source_path, utc_now_iso
from ingestion.parsers import parse_file


@dataclass
class IngestionResult:
    """Documents newly ingested by one pipeline run."""

    documents: list[Document] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None


def default_manifest_path() -> Path:
    settings = get_settings()
    return settings.project_root / "var" / "ingestion" / "manifest.json"


def ingest_file(
    path: Path,
    *,
    corpus_root: Path | None = None,
    manifest_path: Path | None = None,
    force: bool = False,
    pdf_parser: str | None = None,
) -> list[Document]:
    """Ingest one file, returning an empty list when it was already ingested."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot ingest missing file: {path}")

    settings = get_settings()
    parser_name = pdf_parser or settings.pdf_parser
    manifest_location = manifest_path or default_manifest_path()
    manifest = IngestionManifest.load(manifest_location)

    document_hash = sha256_file(path)
    if document_hash in manifest.entries and not force:
        return []

    documents = parse_file(
        path,
        corpus_root=corpus_root,
        document_hash=document_hash,
        pdf_parser=parser_name,
    )

    source_file = relative_source_path(path, corpus_root)
    file_type = path.suffix.lower().lstrip(".")
    if file_type == "markdown":
        file_type = "md"

    ingestion_timestamp = utc_now_iso()
    for document in documents:
        document.metadata.setdefault("source_file", source_file)
        document.metadata.setdefault("file_type", file_type)
        document.metadata["document_hash"] = document_hash
        document.metadata["ingestion_timestamp"] = ingestion_timestamp

    # Ensure required metadata exists even when a parser returns nothing.
    if not documents:
        metadata = base_metadata(
            path,
            corpus_root=corpus_root,
            document_hash=document_hash,
            page_number=None,
            section_heading=None,
        )
        metadata["ingestion_timestamp"] = ingestion_timestamp
        documents = [Document(page_content="", metadata=metadata)]

    manifest.record(
        document_hash,
        {
            "source_file": source_file,
            "file_type": file_type,
            "ingestion_timestamp": ingestion_timestamp,
            "document_count": len(documents),
        },
    )
    manifest.save()
    return documents


def ingest_directory(
    root: Path,
    *,
    manifest_path: Path | None = None,
    force: bool = False,
    pdf_parser: str | None = None,
) -> IngestionResult:
    """Ingest every supported file under ``root``."""

    root = Path(root).resolve()
    manifest_location = manifest_path or default_manifest_path()
    result = IngestionResult(manifest_path=manifest_location)

    for path in discover_documents(root):
        documents = ingest_file(
            path,
            corpus_root=root,
            manifest_path=manifest_location,
            force=force,
            pdf_parser=pdf_parser,
        )
        if documents:
            result.documents.extend(documents)
        else:
            result.skipped.append(path)

    return result
