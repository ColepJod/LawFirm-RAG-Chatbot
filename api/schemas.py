"""Request and response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100000)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    grounding_verdict: str = ""
    thread_id: str


class IngestResponse(BaseModel):
    status: str
    ingested_documents: int = 0
    skipped_documents: int = 0
    manifest_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, Any]


class MetricsResponse(BaseModel):
    request_count: int
    chat_count: int
    ingest_count: int
    error_count: int
    trace_count: int
    last_trace_path: str | None
