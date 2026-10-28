"""Structured JSON tracing for graph runs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import get_settings


@dataclass
class RunTrace:
    """One structured trace for a graph invocation."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    query: str = ""
    standalone_query: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    reranker_scores: list[float] = field(default_factory=list)
    route_decision: str = ""
    grounding_verdict: str = ""
    final_context_length: int = 0
    response: str = ""
    node_latencies_ms: dict[str, float] = field(default_factory=dict)
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceLogger:
    """Append structured JSON traces to a local JSONL file."""

    def __init__(self, settings=None, path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.path = path or (self.settings.trace_dir / "traces.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, trace: RunTrace) -> Path:
        trace.ended_at = datetime.now(UTC).isoformat()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return self.path

    @staticmethod
    def now_ms() -> float:
        return time.perf_counter() * 1000.0
