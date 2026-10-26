"""Context headers prepended to chunks before embedding."""

from __future__ import annotations

from langchain_core.documents import Document


def resolve_section_heading(document: Document, fallback: str | None = None) -> str:
    """Choose the best available section heading for a chunk."""

    if fallback and fallback.strip():
        return fallback.strip()

    metadata = document.metadata or {}
    for key in ("section_heading", "title", "Header 4", "Header 3", "Header 2", "Header 1"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "document"


def build_context_header(document: Document, section_heading: str | None = None) -> str:
    """Build the required ``Document: X | Section: Y`` header."""

    metadata = document.metadata or {}
    source = metadata.get("source_file") or metadata.get("source") or "unknown"
    section = resolve_section_heading(document, section_heading)
    return f"Document: {source} | Section: {section}"
