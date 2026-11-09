"""Incremental, document-aware indexing for dense and sparse retrieval."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from config import get_settings
from ingestion.hashing import sha256_text
from retrieval.bm25 import BM25Index
from retrieval.vectorstores import VectorStoreHandle

SCHEMA_VERSION = 2


class IndexFingerprintMismatch(RuntimeError):
    """Raised when an index was built with incompatible settings."""


@dataclass
class IndexResult:
    """Summary of one indexing run."""

    new_count: int = 0
    skipped_count: int = 0
    removed_count: int = 0
    total_count: int = 0
    fingerprint: str | None = None


@dataclass
class IndexManifest:
    """JSON manifest of indexed chunk IDs, source files, and index settings."""

    path: Path
    fingerprint: str | None = None
    chunks: dict[str, dict[str, object]] = field(default_factory=dict)
    files: dict[str, list[str]] = field(default_factory=dict)

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

        if raw.get("schema_version") == SCHEMA_VERSION:
            chunks = raw.get("chunks", {})
            files = raw.get("files", {})
            return cls(
                path=path,
                fingerprint=raw.get("fingerprint"),
                chunks=dict(chunks) if isinstance(chunks, dict) else {},
                files={
                    str(source): [str(chunk_id) for chunk_id in ids]
                    for source, ids in (files.items() if isinstance(files, dict) else [])
                },
            )

        # Legacy format: a flat mapping of chunk_id -> {"indexed": true}.
        return cls(path=path, chunks=dict(raw), files={})

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    def record(
        self,
        chunk_id: str,
        metadata: dict[str, object] | None = None,
        source_file: str | None = None,
    ) -> None:
        self.chunks[chunk_id] = dict(metadata or {"indexed": True})
        if source_file:
            ids = self.files.setdefault(source_file, [])
            if chunk_id not in ids:
                ids.append(chunk_id)

    def remove_ids(self, chunk_ids: set[str]) -> list[str]:
        removed: list[str] = []
        for chunk_id in chunk_ids:
            if self.chunks.pop(chunk_id, None) is not None:
                removed.append(chunk_id)
        if removed:
            removed_set = set(removed)
            for source, ids in list(self.files.items()):
                remaining = [chunk_id for chunk_id in ids if chunk_id not in removed_set]
                if remaining:
                    self.files[source] = remaining
                else:
                    self.files.pop(source, None)
        return removed

    def rebuild_files(self, documents: list[Document]) -> None:
        """Rebuild the source-file map from the authoritative chunk store."""

        files: dict[str, list[str]] = {}
        for document in documents:
            source = source_file_of(document)
            files.setdefault(source, []).append(chunk_id(document))
        self.files = files

    def all_ids(self) -> list[str]:
        return list(self.chunks)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "chunks": self.chunks,
            "files": self.files,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def chunk_id(document: Document) -> str:
    """Return a stable ID for a chunk."""

    value = (document.metadata or {}).get("chunk_id")
    if isinstance(value, str) and value.strip():
        return value
    return sha256_text(document.page_content)


def source_file_of(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("source_file", "source"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def default_index_dir() -> Path:
    return get_settings().index_dir


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


def _record(document: Document) -> dict[str, object]:
    return {
        "id": chunk_id(document),
        "page_content": document.page_content,
        "metadata": document.metadata or {},
    }


def _rewrite_indexed_chunks(path: Path, chunks: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(_record(chunk), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _deduplicate(chunks: list[Document]) -> list[Document]:
    by_id: dict[str, Document] = {}
    for chunk in chunks:
        by_id[chunk_id(chunk)] = chunk
    return list(by_id.values())


def _check_fingerprint(manifest: IndexManifest, fingerprint: str, *, force: bool) -> bool:
    """Validate the index fingerprint; return True when a full rebuild is needed."""

    if force:
        return True
    if manifest.fingerprint is None:
        manifest.fingerprint = fingerprint
        return False
    if manifest.fingerprint == fingerprint:
        return False
    raise IndexFingerprintMismatch(
        "The index was built with different embedding/chunk settings "
        f"({manifest.fingerprint} != {fingerprint}). Re-run indexing with force=True."
    )


def index_chunks(
    chunks: list[Document],
    vector_store: VectorStoreHandle,
    bm25_index: BM25Index,
    *,
    manifest_path: Path | None = None,
    chunk_store_path: Path | None = None,
    force: bool = False,
    fingerprint: str | None = None,
) -> IndexResult:
    """Index new chunks and replace stale chunks for changed source files.

    A full-file reindex is detected from ``source_file`` metadata: old IDs for
    that source are removed before new chunks are appended. This prevents the
    index from accumulating obsolete versions of a document.
    """

    settings = get_settings()
    manifest_location = manifest_path or default_index_manifest_path()
    store_location = chunk_store_path or default_chunk_store_path()
    current_fingerprint = fingerprint or settings.index_fingerprint

    manifest = IndexManifest.load(manifest_location)
    existing_chunks = load_indexed_chunks(store_location)
    existing_ids = {chunk_id(chunk) for chunk in existing_chunks}
    # The chunk store is authoritative. Drop manifest entries whose payload is
    # no longer present so a missing/partial store cannot silently hide new work.
    for stale_id in [chunk_id for chunk_id in manifest.chunks if chunk_id not in existing_ids]:
        manifest.chunks.pop(stale_id, None)
    manifest.rebuild_files(existing_chunks)

    rebuild_all = _check_fingerprint(manifest, current_fingerprint, force=force)

    removed_ids: set[str] = set()
    if rebuild_all:
        removed_ids.update(manifest.all_ids())
        manifest.chunks.clear()
        manifest.files.clear()
        existing_chunks = []
        new_chunks = list(chunks)
    else:
        new_ids = {chunk_id(chunk) for chunk in chunks}
        incoming_sources = {source_file_of(chunk) for chunk in chunks}
        for source in incoming_sources:
            for old_id in manifest.files.get(source, []):
                if old_id not in new_ids:
                    removed_ids.add(old_id)
        if removed_ids:
            manifest.remove_ids(removed_ids)
            existing_chunks = [
                chunk for chunk in existing_chunks if chunk_id(chunk) not in removed_ids
            ]
        new_chunks = [chunk for chunk in chunks if not manifest.contains(chunk_id(chunk))]

    if removed_ids:
        with contextlib.suppress(Exception):
            vector_store.delete(list(removed_ids))

    combined = _deduplicate([*existing_chunks, *new_chunks])
    _rewrite_indexed_chunks(store_location, combined)

    if new_chunks:
        vector_store.add_documents(new_chunks, ids=[chunk_id(chunk) for chunk in new_chunks])

    bm25_index.replace_documents(combined)

    for chunk in new_chunks:
        manifest.record(
            chunk_id(chunk),
            {"indexed": True, "source_file": source_file_of(chunk)},
            source_file=source_file_of(chunk),
        )
    manifest.fingerprint = current_fingerprint
    manifest.save()

    return IndexResult(
        new_count=len(new_chunks),
        skipped_count=len(chunks) - len(new_chunks),
        removed_count=len(removed_ids),
        total_count=len(combined),
        fingerprint=current_fingerprint,
    )


def remove_source(
    source_file: str,
    vector_store: VectorStoreHandle,
    bm25_index: BM25Index,
    *,
    manifest_path: Path | None = None,
    chunk_store_path: Path | None = None,
) -> int:
    """Remove every indexed chunk that belongs to ``source_file``.

    Returns the number of removed chunks. The chunk store stays authoritative:
    only IDs present in the store are deleted from the vector database.
    """

    manifest_location = manifest_path or default_index_manifest_path()
    store_location = chunk_store_path or default_chunk_store_path()
    manifest = IndexManifest.load(manifest_location)
    existing_chunks = load_indexed_chunks(store_location)

    existing_ids = {chunk_id(chunk) for chunk in existing_chunks}
    removed_ids = {
        chunk_id(chunk) for chunk in existing_chunks if source_file_of(chunk) == source_file
    }
    removed_ids.update(
        chunk_id for chunk_id in manifest.files.get(source_file, []) if chunk_id in existing_ids
    )
    if not removed_ids:
        return 0

    remaining = [chunk for chunk in existing_chunks if chunk_id(chunk) not in removed_ids]
    with contextlib.suppress(Exception):
        vector_store.delete(list(removed_ids))

    _rewrite_indexed_chunks(store_location, remaining)
    bm25_index.replace_documents(remaining)
    manifest.remove_ids(removed_ids)
    manifest.rebuild_files(remaining)
    manifest.save()
    return len(removed_ids)
