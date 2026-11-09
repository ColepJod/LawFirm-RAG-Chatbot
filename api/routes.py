"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
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
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

from api.factory import rebuild_retriever
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    JobResponse,
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
from api.state import JOB_TTL_SECONDS, AppState
from chunking import chunk_documents
from ingestion import IngestionManifest, ingest_directory, ingest_file
from retrieval import index_chunks, load_indexed_chunks, remove_source, source_file_of

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


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


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


def _prepare_chat(state: AppState, payload: ChatRequest, request: Request) -> tuple[str, str]:
    """Run shared validation for chat and streaming chat."""

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
    return request_id, conversation_id


def _build_chat_response(
    state: AppState,
    result: dict,
    request_id: str,
    conversation_id: str,
    started: float,
) -> ChatResponse:
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


def _graph_config(state: AppState, conversation_id: str) -> dict:
    return {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": state.settings.max_graph_steps,
    }


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

    caches: dict[str, object] = {}
    for name, component in (
        ("retrieval", state.dependencies.retriever if state.dependencies else None),
        ("reranker", state.dependencies.reranker if state.dependencies else None),
    ):
        stats = getattr(component, "stats", None)
        if callable(stats):
            try:
                caches[name] = stats()
            except Exception:
                logger.debug("failed to read %s cache stats", name, exc_info=True)

    return MetricsResponse(
        request_count=state.request_count,
        chat_count=state.chat_count,
        ingest_count=state.ingest_count,
        error_count=state.error_count,
        trace_count=trace_count,
        last_trace_path=str(state.trace_path) if state.trace_path else None,
        caches=caches,
    )


def _prune_jobs(state: AppState) -> None:
    now = time.time()
    stale = [job_id for job_id, job in state.jobs.items() if now - job.updated_at > JOB_TTL_SECONDS]
    for job_id in stale:
        state.jobs.pop(job_id, None)


def _create_job(state: AppState, kind: str) -> str:
    job_id = uuid.uuid4().hex
    from api.state import JobRecord

    with state.jobs_lock:
        _prune_jobs(state)
        state.jobs[job_id] = JobRecord(job_id=job_id, kind=kind)
    return job_id


def _update_job(state: AppState, job_id: str, **changes: object) -> None:
    with state.jobs_lock:
        job = state.jobs.get(job_id)
        if job is not None:
            job.update(**changes)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def job_status(job_id: str, request: Request, state: AppState = Depends(get_state)):
    require_admin(request, state.settings)
    with state.jobs_lock:
        job = state.jobs.get(job_id)
        payload = job.to_dict() if job is not None else None
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "message": f"Unknown job: {job_id}"},
        )
    return JobResponse.model_validate(payload)


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
            detail={
                "code": "missing_input",
                "message": "Provide either a file or a directory",
            },
        )

    job_id = _create_job(state, "ingest")
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
        background_tasks.add_task(
            _ingest_path_task, state, job_id, destination, destination.parent, force
        )
        return IngestResponse(
            status="queued",
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
        _ingest_path_task, state, job_id, resolved_directory, resolved_directory, force
    )
    return IngestResponse(
        status="queued",
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
    job_id: str,
    path: Path,
    corpus_root: Path,
    force: bool = False,
) -> None:
    """Background ingestion job shared by uploads and directory ingestion."""

    _update_job(state, job_id, status="running")
    try:
        if path.is_dir():
            result = ingest_directory(
                path,
                manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
                force=force,
            )
            documents = result.documents
            skipped = len(result.skipped)
        else:
            documents = ingest_file(
                path,
                corpus_root=corpus_root,
                manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
                force=force,
            )
            skipped = 0 if documents else 1
        indexed = _index_documents(state, documents, force=force)
        _update_job(
            state,
            job_id,
            status="succeeded",
            documents=len(documents),
            chunks=indexed,
            message=f"indexed {indexed} chunks; skipped {skipped}",
        )
        logger.info(
            "ingestion job finished job=%s path=%s documents=%d indexed_chunks=%d",
            job_id,
            path,
            len(documents),
            indexed,
        )
    except Exception as exc:
        state.error_count += 1
        _update_job(state, job_id, status="failed", error=str(exc))
        logger.exception("ingestion job failed job=%s path=%s", job_id, path)


def _remove_document(state: AppState, source_file: str) -> int:
    """Delete one source from the index and ingestion manifests."""

    with state.index_lock:
        removed = remove_source(
            source_file,
            state.vector_store,
            state.bm25_index,
            manifest_path=state.settings.index_dir / "manifest.json",
            chunk_store_path=state.settings.index_dir / "chunks.jsonl",
        )
        if not removed:
            return 0
        ingestion_manifest = IngestionManifest.load(
            state.settings.project_root / "var" / "ingestion" / "manifest.json"
        )
        if ingestion_manifest.remove_by_source(source_file):
            ingestion_manifest.save()
        rebuild_retriever(state)
    return removed


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(request: Request, state: AppState = Depends(get_state)):
    require_admin(request, state.settings)
    chunks = await run_in_threadpool(load_indexed_chunks, state.settings.index_dir / "chunks.jsonl")
    counts = Counter(source_file_of(chunk) for chunk in chunks)
    return DocumentsResponse(
        documents=[
            {"source_file": source_file, "chunks": count}
            for source_file, count in sorted(counts.items())
        ],
        total_chunks=len(chunks),
    )


@router.delete("/documents/{source_file:path}", response_model=DeleteDocumentResponse)
async def delete_document(source_file: str, request: Request, state: AppState = Depends(get_state)):
    require_admin(request, state.settings)
    normalized = source_file.strip("/")
    removed = await run_in_threadpool(_remove_document, state, normalized)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "document_not_found",
                "message": f"No indexed chunks for source: {normalized}",
            },
        )
    return DeleteDocumentResponse(status="deleted", source_file=normalized, removed_chunks=removed)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    state: AppState = Depends(get_state),
):
    state.request_count += 1
    started = time.perf_counter()
    request_id, conversation_id = _prepare_chat(state, payload, request)

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

    return _build_chat_response(state, result, request_id, conversation_id, started)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    state: AppState = Depends(get_state),
):
    if not state.settings.chat_stream_enabled:
        raise HTTPException(
            status_code=404,
            detail={"code": "streaming_disabled", "message": "Streaming is disabled."},
        )

    state.request_count += 1
    started = time.perf_counter()
    request_id, conversation_id = _prepare_chat(state, payload, request)
    config = _graph_config(state, conversation_id)
    graph_input = {"messages": [HumanMessage(content=payload.query)]}

    async def event_stream():
        yield _sse(
            "meta",
            {"request_id": request_id, "thread_id": conversation_id},
        )
        accumulated: dict = {}
        try:
            async for chunk in state.graph.astream(
                graph_input, config=config, stream_mode="updates"
            ):
                if not isinstance(chunk, dict):
                    continue
                for node, update in chunk.items():
                    if isinstance(update, dict):
                        accumulated.update(update)
                    latency = None
                    if isinstance(update, dict):
                        latency = (update.get("node_latencies") or {}).get(node)
                    yield _sse(
                        "progress",
                        {"node": node, "latency_ms": latency},
                    )
            response = _build_chat_response(
                state, accumulated, request_id, conversation_id, started
            )
            yield _sse("done", response.model_dump())
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
            yield _sse("error", detail)
        except Exception:
            state.error_count += 1
            logger.exception("streaming chat failed request_id=%s", request_id)
            yield _sse(
                "error",
                {
                    "code": "chat_failed",
                    "message": "The chat request failed.",
                    "request_id": request_id,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
