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

JOB_TTL_SECONDS = 3600.0


@dataclass
class JobRecord:
    """In-memory status for one ingestion job."""

    job_id: str
    kind: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    documents: int = 0
    chunks: int = 0
    message: str | None = None
    error: str | None = None

    def update(self, **changes: object) -> None:
        for key, value in changes.items():
            setattr(self, key, value)
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "documents": self.documents,
            "chunks": self.chunks,
            "message": self.message,
            "error": self.error,
        }


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
    jobs: dict[str, JobRecord] = field(default_factory=dict)
    jobs_lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.time)
