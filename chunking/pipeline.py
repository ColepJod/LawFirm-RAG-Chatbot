"""Chunking entry points used by ingestion and retrieval preparation."""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.recursive import RecursiveTokenChunker
from chunking.structure import StructureAwareChunker
from config import get_settings


def get_chunker(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_chars: int | None = None,
):
    """Create a chunker from explicit values or application settings."""

    settings = get_settings()
    resolved_strategy = strategy or settings.chunk_strategy
    resolved_size = chunk_size if chunk_size is not None else settings.chunk_size
    resolved_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    resolved_min_chars = (
        min_chunk_chars if min_chunk_chars is not None else settings.min_chunk_chars
    )

    if resolved_strategy == "recursive":
        return RecursiveTokenChunker(
            chunk_size=resolved_size,
            chunk_overlap=resolved_overlap,
            min_chunk_chars=resolved_min_chars,
        )
    if resolved_strategy == "structure":
        return StructureAwareChunker(
            chunk_size=resolved_size,
            chunk_overlap=resolved_overlap,
            min_chunk_chars=resolved_min_chars,
        )

    raise ValueError(f"Unsupported chunking strategy: {resolved_strategy}")


def chunk_documents(
    documents: list[Document],
    *,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_chars: int | None = None,
) -> list[Document]:
    """Chunk documents using the configured strategy."""

    chunker = get_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )
    return chunker.split_documents(documents)
