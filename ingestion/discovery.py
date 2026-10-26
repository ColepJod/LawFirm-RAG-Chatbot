"""File discovery for supported corpus formats."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".markdown", ".txt", ".docx"})


def discover_documents(root: Path) -> list[Path]:
    """Return supported files under ``root`` in deterministic order."""

    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
