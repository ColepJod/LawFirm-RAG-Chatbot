"""Incremental indexing for dense and sparse retrieval."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from config import get_settings
from ingestion.hashing import sha256_text
from retrieval.bm25 import BM25Index
from retrieval.vectorstores import VectorStoreHandle


@dataclass
class IndexResult:
    """Summary of one indexing run."""

    new_count: int = 0
    skipped_count: int = 0
    total_count: int = 0


@dataclass
class IndexManifest:
    """JSON manifest of indexed chunk IDs."""

    path: Path
    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> IndexManifest:
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read index manifest {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Index manifest must be a JSON object: {path}")
        return cls(path=path, entries=raw)

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self.entries

    def record(self, chunk_id: str, metadata: dict[str, object]) -> None:
        self.entries[chunk_id] = metadata

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        import os

        os.replace(temporary, self.path)


def chunk_id(document: Document) -> str:
    """Return a stable ID for a chunk."""

    value = (document.metadata or {}).get("chunk_id")
    if isinstance(value, str) and value.strip():
        return value
    return sha256_text(document.page_content)


def default_index_dir() -> Path:
    return get_settings().project_root / "var" / "index"


def default_index_manifest_path() -> Path:
    return default_index_dir() / "manifest.json"


def default_chunk_store_path() -> Path:
    return default_index_dir() / "chunks.jsonl"


def load_indexed_chunks(path: Path) -> list[Document]:
    """Load previously indexed chunks from JSONL."""

    if not path.exists():
        return []

    documents: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        documents.append(
            Document(
                page_content=record["page_content"],
                metadata=dict(record.get("metadata", {})),
            )
        )
    return documents


def _append_indexed_chunks(path: Path, chunks: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for chunk in chunks:
            record = {
                "id": chunk_id(chunk),
                "page_content": chunk.page_content,
                "metadata": chunk.metadata or {},
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_indexed_chunks(path: Path, chunks: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            record = {
                "id": chunk_id(chunk),
                "page_content": chunk.page_content,
                "metadata": chunk.metadata or {},
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def index_chunks(
    chunks: list[Document],
    vector_store: VectorStoreHandle,
    bm25_index: BM25Index,
    *,
    manifest_path: Path | None = None,
    chunk_store_path: Path | None = None,
    force: bool = False,
) -> IndexResult:
    """Index only chunks that have not been indexed before."""

    manifest_location = manifest_path or default_index_manifest_path()
    store_location = chunk_store_path or default_chunk_store_path()
    manifest = IndexManifest.load(manifest_location)

    if force:
        existing_ids = list(manifest.entries)
        if existing_ids:
            with contextlib.suppress(Exception):
                vector_store.delete(existing_ids)
        _rewrite_indexed_chunks(store_location, chunks)
        new_chunks = list(chunks)
        manifest.entries.clear()
    else:
        new_chunks = [chunk for chunk in chunks if chunk_id(chunk) not in manifest.entries]
        if new_chunks:
            _append_indexed_chunks(store_location, new_chunks)

    if new_chunks:
        vector_store.add_documents(new_chunks, ids=[chunk_id(chunk) for chunk in new_chunks])

    all_chunks = load_indexed_chunks(store_location)
    bm25_index.replace_documents(all_chunks)

    for chunk in new_chunks:
        manifest.record(chunk_id(chunk), {"indexed": True})

    manifest.save()
    return IndexResult(
        new_count=len(new_chunks),
        skipped_count=len(chunks) - len(new_chunks),
        total_count=len(all_chunks),
    )
