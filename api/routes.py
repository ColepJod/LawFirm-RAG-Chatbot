"""FastAPI route handlers."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from starlette.concurrency import run_in_threadpool

from api.factory import rebuild_retriever
from api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse, MetricsResponse
from api.state import AppState
from chunking import chunk_documents
from ingestion import ingest_directory, ingest_file
from retrieval import index_chunks

router = APIRouter()


def get_state() -> AppState:
    from api.main import app_state

    if app_state is None:
        raise HTTPException(status_code=503, detail="Application is not initialized")
    return app_state


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
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    directory: str | None = Form(default=None),
    state: AppState = Depends(get_state),
):
    state.request_count += 1
    state.ingest_count += 1

    if file is None and not directory:
        raise HTTPException(status_code=400, detail="Provide either a file or a directory")

    if file is not None:
        state.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = state.settings.upload_dir / (file.filename or "uploaded_document")
        with destination.open("wb") as stream:
            shutil.copyfileobj(file.file, stream)
        background_tasks.add_task(_ingest_file_task, state, destination)
        return IngestResponse(status="accepted", ingested_documents=0, skipped_documents=0)

    resolved_directory = Path(directory)
    if not resolved_directory.is_absolute():
        resolved_directory = state.settings.project_root / resolved_directory
    if not resolved_directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {resolved_directory}")
    background_tasks.add_task(_ingest_directory_task, state, resolved_directory)
    return IngestResponse(status="accepted", ingested_documents=0, skipped_documents=0)


def _index_documents(state: AppState, documents) -> None:
    chunks = chunk_documents(documents)
    if not chunks:
        return
    index_chunks(
        chunks,
        state.vector_store,
        state.bm25_index,
        manifest_path=state.settings.project_root / "var" / "index" / "manifest.json",
        chunk_store_path=state.settings.project_root / "var" / "index" / "chunks.jsonl",
    )
    rebuild_retriever(state)


def _ingest_file_task(state: AppState, path: Path) -> None:
    try:
        documents = ingest_file(
            path,
            manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
        )
        _index_documents(state, documents)
    except Exception:
        state.error_count += 1
        raise


def _ingest_directory_task(state: AppState, path: Path) -> None:
    try:
        result = ingest_directory(
            path,
            manifest_path=state.settings.project_root / "var" / "ingestion" / "manifest.json",
        )
        _index_documents(state, result.documents)
    except Exception:
        state.error_count += 1
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, state: AppState = Depends(get_state)):
    state.request_count += 1
    state.chat_count += 1

    validation = state.input_guardrails.validate(request.query)
    if not validation.allowed:
        state.error_count += 1
        raise HTTPException(status_code=400, detail={"reasons": validation.reasons})

    thread_id = request.thread_id or "default"

    try:
        result = await run_in_threadpool(_run_graph, state, request.query, thread_id)
    except Exception as exc:
        state.error_count += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = str(result.get("generation", ""))
    citations = list(result.get("citations", []))
    retrieved_source_ids = list(result.get("retrieved_source_ids", []))
    grounding_verdict = str(result.get("grounding_verdict", ""))

    documents = list(result.get("reranked_chunks", []) or result.get("retrieved_chunks", []) or [])
    route_decision = str(result.get("route_decision") or "")
    expected_context = route_decision == "rag" or bool(documents)

    output_validation = state.output_guardrails.validate_citations(answer, documents)
    refusal_validation = state.output_guardrails.validate_refusal(
        answer, documents, expect_context=expected_context
    )
    if not output_validation.allowed or not refusal_validation.allowed:
        state.error_count += 1
        raise HTTPException(
            status_code=422,
            detail={
                "reasons": output_validation.reasons + refusal_validation.reasons,
                "answer": answer,
            },
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        retrieved_source_ids=retrieved_source_ids,
        grounding_verdict=grounding_verdict,
        thread_id=thread_id,
    )


def _run_graph(state: AppState, query: str, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": state.settings.max_graph_steps,
    }
    return state.graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)
