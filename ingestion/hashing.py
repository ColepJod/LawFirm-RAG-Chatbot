"""Content hashing for idempotent ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 hex digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return a stable SHA-256 hex digest for decoded text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
