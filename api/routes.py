"""FastAPI route handlers."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

from api.factory import rebuild_retriever
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    Source,
)
from api.security import (
    client_key,
    normalize_conversation_id,
    require_admin,
    sanitize_filename,
    save_upload_limited,
)
from api.state import AppState
from chunking import chunk_documents
from ingestion import ingest_directory, ingest_file
from retrieval import index_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


def get_state() -> AppState:
    from api.main import app_state

    if app_state is None:
        raise HTTPException(status_code=503, detail="Application is not initialized")
    return app_state


def _request_id(request: Request) -> str:
    supplied = (request.headers.get("x-request-id") or "").strip()
    if supplied and len(supplied) <= 128 and supplied.isprintable():
        return supplied
    return uuid.uuid4().hex


def _enforce_rate_limit(state: AppState, request: Request) -> None:
    limiter = state.rate_limiter
    if limiter is None:
        return
    if not limiter.allow(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limit_exceeded",
                "message": "Too many requests. Please retry shortly.",
            },
        )


def _ensure_index_ready(state: AppState) -> None:
    """Refuse to serve answers from an index built with other settings."""

    if state.index_fingerprint_ok:
        return
    if state.settings.index_fingerprint_mode != "enforce":
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": "index_fingerprint_mismatch",
            "message": (
                "The vector index was built with different embedding or chunking settings. "
                "Re-run ingestion to rebuild it before chatting."
            ),
            "expected": state.settings.index_fingerprint,
            "found": state.index_manifest.fingerprint if state.index_manifest else None,
        },
    )


def _track_conversation(state: AppState, conversation_id: str) -> None:
    """Record activity and evict conversations past the checkpoint TTL."""

    now = time.time()
    ttl = state.settings.checkpoint_ttl_seconds
    with state.conversation_lock:
        state.conversation_last_seen[conversation_id] = now
        if len(state.conversation_last_seen) < 2:
            return
        stale = [key for key, seen in state.conversation_last_seen.items() if now - seen > ttl]
        for key in stale:
            state.conversation_last_seen.pop(key, None)


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)):
    from observability import health_check

    state.request_count += 1
    report = await run_in_threadpool(
        health_check,
        state.settings,
        state.vector_store,
        state.bm25_index,
    )
    report["checks"]["index"] = {
        "status": "ok" if state.index_fingerprint_ok else "fingerprint_mismatch",
        "fingerprint": state.index_fingerprint,
        "mode": state.settings.index_fingerprint_mode,
        "chunk_count": state.bm25_index.document_count(),
    }
    return HealthResponse(status=report["status"], checks=report["checks"])


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(state: AppState = Depends(get_state)):
    state.request_count += 1
    trace_count = 0
    if state.trace_path is not None and state.trace_path.exists():
        trace_count = sum(1 for _ in state.trace_path.open(encoding="utf-8"))
    return MetricsResponse(
        request_count=state.request_count,
        chat_count=state.chat_count,
        ingest_count=state.ingest_count,
        error_count=state.error_count,
        trace_count=trace_count,
        last_trace_path=str(state.trace_path) if state.trace_path else None,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    directory: str | None = Form(default=None),
    force: bool = Form(default=False),
    state: AppState = Depends(get_state),
):
    state.request_count += 1
    require_admin(request, state.settings)
    _enforce_rate_limit(state, request)

    if file is None and not directory:
        raise HTTPException(
            status_code=400,
            detail={"code": "missing_input", "message": "Provide either a file or a directory"},
        )

    job_id = uuid.uuid4().hex
    if file is not None:
        safe_name = sanitize_filename(file.filename)
        # Give each upload its own folder so concurrent uploads cannot overwrite
        # each other, while the ingested ``source_file`` stays the clean name.
        destination = state.settings.upload_dir / job_id / safe_name
        await save_upload_limited(
            file,
            destination,
            max_bytes=state.settings.max_upload_mb * 1024 * 1024,
        )
        state.ingest_count += 1
        background_tasks.add_task(_ingest_path_task, state, destination, destination.parent, force)
        return IngestResponse(
            status="accepted",
            job_id=job_id,
            message=f"Upload accepted and queued for indexing ({safe_name}).",
        )

    resolved_directory = Path(directory or "").expanduser()
    if not resolved_directory.is_absolute():
        resolved_directory = state.settings.project_root / resolved_directory
    resolved_directory = resolved_directory.resolve()
    if not resolved_directory.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "directory_not_found",
                "message": f"Directory not found: {resolved_directory}",
            },
        )
    if not resolved_directory.is_dir():
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_directory", "message": "Path is not a directory"},
        )

    state.ingest_count += 1
    background_tasks.add_task(
        _ingest_path_task, state, resolved_directory, resolved_directory, force
    )
    return IngestResponse(
        status="accepted",
        job_id=job_id,
        message=f"Directory queued for indexing: {resolved_directory}",
    )


def _index_documents(state: AppState, documents, *, force: bool = False) -> int:
    """Chunk, index, and publish a new retriever under the index lock."""

    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    with state.index_lock:
        result = index_chunks(
            chunks,
            state.vector_store,
            state.bm25_index,
            manifest_path=state.settings.index_dir / "manifest.json",
            chunk_store_path=state.settings.index_dir / "chunks.jsonl",
            force=force or not state.index_fingerprint_ok,
        )
        rebuild_retriever(state)
    return result.new_count


def _ingest_path_task(
    state: AppState,
    path: Path,
    corpus_root: Path,
    force: bool = False,
) -> None:
    """Background ingestion job shared by uploads and directory ingestion."""

    try:
        if path.is_dir():
            result = ingest_directory(
                path,
                manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
                force=force,
            )
            documents = result.documents
        else:
            documents = ingest_file(
                path,
                corpus_root=corpus_root,
                manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
                force=force,
            )
        indexed = _index_documents(state, documents, force=force)
        logger.info(
            "ingestion job finished path=%s documents=%d indexed_chunks=%d",
            path,
            len(documents),
            indexed,
        )
    except Exception:
        state.error_count += 1
        logger.exception("ingestion job failed path=%s", path)
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    state: AppState = Depends(get_state),
):
    state.request_count += 1
    started = time.perf_counter()
    request_id = _request_id(request)
    _enforce_rate_limit(state, request)

    validation = state.input_guardrails.validate(payload.query)
    if not validation.allowed:
        state.error_count += 1
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_input",
                "message": "The query was rejected by input guardrails.",
                "reasons": validation.reasons,
            },
        )

    _ensure_index_ready(state)

    conversation_id = normalize_conversation_id(payload.conversation_id or payload.thread_id)
    _track_conversation(state, conversation_id)
    state.chat_count += 1

    try:
        result = await run_in_threadpool(_run_graph, state, payload.query, conversation_id)
    except Exception as exc:
        state.error_count += 1
        logger.exception("chat request failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "chat_failed",
                "message": "The chat request failed.",
                "request_id": request_id,
            },
        ) from exc

    answer = str(result.get("generation", ""))
    citations = list(result.get("citations", []))
    retrieved_source_ids = list(result.get("retrieved_source_ids", []))
    grounding_verdict = str(result.get("grounding_verdict", ""))
    warnings = list(result.get("citation_warnings", []) or [])
    sources = _coerce_sources(result.get("sources", []))

    documents = list(result.get("reranked_chunks", []) or result.get("retrieved_chunks", []) or [])
    route_decision = str(result.get("route_decision") or "")
    expected_context = route_decision == "rag" or bool(documents)

    output_validation = state.output_guardrails.validate_citations(answer, documents)
    if not output_validation.allowed:
        # A missing citation marker is a quality warning, not a failed request:
        # the answer is still useful and the UI can show unranked sources.
        warnings.extend(output_validation.reasons)

    refusal_validation = state.output_guardrails.validate_refusal(
        answer, documents, expect_context=expected_context
    )
    if not refusal_validation.allowed:
        state.error_count += 1
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ungrounded_answer",
                "message": "The answer was generated without supporting context.",
                "reasons": refusal_validation.reasons,
                "request_id": request_id,
            },
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        sources=sources,
        warnings=warnings,
        retrieved_source_ids=retrieved_source_ids,
        grounding_verdict=grounding_verdict,
        thread_id=conversation_id,
        request_id=request_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        trace_id=conversation_id,
    )


def _coerce_sources(raw_sources) -> list[Source]:
    sources: list[Source] = []
    for raw in raw_sources or []:
        try:
            sources.append(Source.model_validate(raw))
        except Exception:
            continue
    return sources


def _run_graph(state: AppState, query: str, conversation_id: str):
    config = {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": state.settings.max_graph_steps,
    }
    return state.graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
