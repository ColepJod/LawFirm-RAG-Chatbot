"""Request and response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100000)
    thread_id: str | None = None
    conversation_id: str | None = None


class Source(BaseModel):
    id: int
    source_file: str
    section: str = "document"
    page: int | None = None
    chunk_id: str = ""
    score: float | None = None
    snippet: str = ""


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    grounding_verdict: str = ""
    thread_id: str
    request_id: str = ""
    latency_ms: float = 0.0
    trace_id: str | None = None


class IngestResponse(BaseModel):
    status: str
    ingested_documents: int = 0
    skipped_documents: int = 0
    manifest_path: str | None = None
    job_id: str | None = None
    message: str | None = None


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str = ""
    details: Any = None


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: float
    updated_at: float
    documents: int = 0
    chunks: int = 0
    message: str | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    source_file: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list)
    total_chunks: int = 0


class DeleteDocumentResponse(BaseModel):
    status: str
    source_file: str
    removed_chunks: int


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
    caches: dict[str, Any] = Field(default_factory=dict)
