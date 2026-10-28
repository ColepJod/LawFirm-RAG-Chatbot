"""Shared application state for the FastAPI layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings
from graph import GraphDependencies
from observability import InputGuardrails, OutputGuardrails, TraceLogger
from retrieval import (
    BM25Index,
)


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
