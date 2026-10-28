"""Golden-set loading and validation for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenItem:
    """One evaluation question with optional expected answer and sources."""

    question: str
    expected_answer: str | None = None
    relevant_source_ids: list[str] = field(default_factory=list)
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Load a JSONL golden set."""

    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    items: list[GoldenItem] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(record, dict) or not record.get("question"):
            raise ValueError(f"Golden item on line {line_number} must contain a question")
        items.append(
            GoldenItem(
                question=str(record["question"]),
                expected_answer=record.get("expected_answer"),
                relevant_source_ids=list(record.get("relevant_source_ids", [])),
                category=str(record.get("category", "general")),
                metadata=dict(record.get("metadata", {})),
            )
        )

    if not items:
        raise ValueError(f"Golden set is empty: {path}")
    return items
