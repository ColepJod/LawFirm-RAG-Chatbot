"""Metadata helpers for ingested documents."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(UTC).isoformat()


def relative_source_path(path: Path, corpus_root: Path | None = None) -> str:
    """Return a portable, forward-slash source path."""

    if corpus_root is not None:
        try:
            return path.relative_to(corpus_root).as_posix()
        except ValueError:
            pass
    return path.name


def base_metadata(
    path: Path,
    *,
    corpus_root: Path | None = None,
    document_hash: str,
    page_number: int | None = None,
    section_heading: str | None = None,
) -> dict[str, object]:
    """Build the required metadata for one source document."""

    suffix = path.suffix.lower().lstrip(".")
    file_type = "md" if suffix == "markdown" else suffix

    return {
        "source_file": relative_source_path(path, corpus_root),
        "file_type": file_type,
        "page_number": page_number,
        "section_heading": section_heading,
        "ingestion_timestamp": utc_now_iso(),
        "document_hash": document_hash,
    }
