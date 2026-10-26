"""Multi-format document ingestion pipeline (Step 2)."""

from ingestion.discovery import SUPPORTED_EXTENSIONS, discover_documents
from ingestion.hashing import sha256_file, sha256_text
from ingestion.manifest import IngestionManifest
from ingestion.metadata import base_metadata, relative_source_path, utc_now_iso
from ingestion.parsers import (
    parse_docx_file,
    parse_file,
    parse_markdown_file,
    parse_pdf_file,
    parse_text_file,
)
from ingestion.pipeline import IngestionResult, default_manifest_path, ingest_directory, ingest_file

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "IngestionManifest",
    "IngestionResult",
    "base_metadata",
    "default_manifest_path",
    "discover_documents",
    "ingest_directory",
    "ingest_file",
    "parse_docx_file",
    "parse_file",
    "parse_markdown_file",
    "parse_pdf_file",
    "parse_text_file",
    "relative_source_path",
    "sha256_file",
    "sha256_text",
    "utc_now_iso",
]
