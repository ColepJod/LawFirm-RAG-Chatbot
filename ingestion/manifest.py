"""Persistent manifest used to make ingestion idempotent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestionManifest:
    """A JSON-backed map from document hash to ingestion metadata."""

    path: Path
    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> IngestionManifest:
        if not path.exists():
            return cls(path=path)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read ingestion manifest {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"Ingestion manifest must be a JSON object: {path}")

        return cls(path=path, entries=raw)

    def contains(self, document_hash: str) -> bool:
        return document_hash in self.entries

    def record(self, document_hash: str, metadata: dict[str, object]) -> None:
        self.entries[document_hash] = metadata

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
