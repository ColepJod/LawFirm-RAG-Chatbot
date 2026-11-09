"""Shared application state for the FastAPI layer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Settings
from graph import GraphDependencies
from observability import InputGuardrails, OutputGuardrails, TraceLogger
from retrieval import BM25Index


@dataclass
class AppState:
    settings: Settings
    graph: Any
    dependencies: GraphDependencies
    vector_store: Any
    bm25_index: BM25Index
    trace_logger: TraceLogger
    input_guardrails: InputGuardrails
    output_guardrails: OutputGuardrails
    request_count: int = 0
    chat_count: int = 0
    ingest_count: int = 0
    error_count: int = 0
    trace_path: Path | None = None
    checkpointer: Any = None
    index_manifest: Any = None
    index_fingerprint: str | None = None
    index_fingerprint_ok: bool = True
    index_lock: threading.RLock = field(default_factory=threading.RLock)
    rate_limiter: Any = None
    conversation_last_seen: dict[str, float] = field(default_factory=dict)
    conversation_lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.time)
