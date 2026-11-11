from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.main import app
from api.security import TokenBucketRateLimiter, sanitize_filename
from api.state import AppState
from config import Settings


class FakeVectorStore:
    def similarity_search(self, query, k=1):
        return []


class FakeBM25:
    def document_count(self):
        return 3


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        return self.result

    def stream(self, payload, config=None, stream_mode="updates"):
        self.calls.append((payload, config))
        yield {"route": {"route_decision": "rag", "node_latencies": {"route": 1.0}}}
        yield {"generate": self.result}


def make_state(tmp_path: Path, **overrides) -> AppState:
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        max_query_chars=50,
        **overrides,
    )
    from observability import InputGuardrails, OutputGuardrails, TraceLogger

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("", encoding="utf-8")
    reranked = Document(
        page_content="Consultations cost $300.",
        metadata={
            "source_file": "rates.md",
            "section_heading": "Billing",
            "reranker_score": 2.5,
        },
    )
    graph = FakeGraph(
        {
            "generation": "Consultations cost $300 [1].",
            "citations": ["rates.md | Billing | p. n/a"],
            "sources": [
                {
                    "id": 1,
                    "source_file": "rates.md",
                    "section": "Billing",
                    "page": None,
                    "chunk_id": "chunk-1",
                    "score": 2.5,
                    "snippet": "Consultations cost $300.",
                }
            ],
            "warnings": [],
            "citation_warnings": [],
            "retrieved_source_ids": ["chunk-1"],
            "grounding_verdict": "pass",
            "reranked_chunks": [reranked],
            "retrieved_chunks": [reranked],
        }
    )
    return AppState(
        settings=settings,
        graph=graph,
        dependencies=None,
        vector_store=FakeVectorStore(),
        bm25_index=FakeBM25(),
        trace_logger=TraceLogger(settings, path=trace_path),
        input_guardrails=InputGuardrails(settings),
        output_guardrails=OutputGuardrails(settings),
        trace_path=trace_path,
        rate_limiter=TokenBucketRateLimiter(settings.rate_limit_per_minute),
    )


def client_for(state: AppState) -> TestClient:
    import api.main as main_module

    main_module.build_app_state = lambda: state
    main_module.app_state = state
    return TestClient(app)


def test_health_endpoint(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "checks" in body
    assert body["checks"]["index"]["status"] == "ok"


def test_metrics_endpoint(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["trace_count"] == 0


def test_chat_endpoint_returns_answer_and_sources(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": "rate?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Consultations cost $300 [1]."
    assert body["sources"][0]["source_file"] == "rates.md"
    assert body["sources"][0]["section"] == "Billing"
    assert body["request_id"]
    assert body["latency_ms"] >= 0


def test_chat_endpoint_assigns_unique_conversation_ids(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    first = client.post("/chat", json={"query": "rate?"}).json()
    second = client.post("/chat", json={"query": "rate?"}).json()

    assert first["thread_id"] != second["thread_id"]
    assert first["thread_id"] != "default"


def test_chat_endpoint_preserves_explicit_conversation_id(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post(
        "/chat",
        json={"query": "rate?", "conversation_id": "client:abc-123"},
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == "client:abc-123"
    assert state.chat_count == 1


def test_chat_endpoint_rejects_invalid_conversation_id(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post(
        "/chat",
        json={"query": "rate?", "conversation_id": "bad id!"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_conversation_id"


def test_chat_endpoint_rejects_long_query(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": "x" * 60})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_input"


def test_chat_endpoint_rejects_empty_query(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": ""})

    assert response.status_code == 422


def test_chat_endpoint_rate_limits_clients(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.rate_limiter = TokenBucketRateLimiter(1)

    client = client_for(state)
    assert client.post("/chat", json={"query": "rate?"}).status_code == 200

    limited = client.post("/chat", json={"query": "rate?"})
    assert limited.status_code == 429
    assert limited.json()["detail"]["code"] == "rate_limit_exceeded"


def test_chat_endpoint_blocks_on_index_fingerprint_mismatch(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.index_fingerprint_ok = False
    state.index_manifest = SimpleNamespace(fingerprint="old-fingerprint")

    client = client_for(state)
    response = client.post("/chat", json={"query": "rate?"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "index_fingerprint_mismatch"


def test_chat_endpoint_warns_when_answer_has_no_citation(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.graph.result["generation"] = "Consultations cost $300."
    state.graph.result["sources"] = []

    client = client_for(state)
    response = client.post("/chat", json={"query": "rate?"})

    assert response.status_code == 200
    assert response.json()["warnings"]


def test_chat_stream_emits_progress_and_done(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    with client.stream("POST", "/chat/stream", json={"query": "rate?"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta" in body
    assert "event: progress" in body
    assert "event: done" in body
    assert "Consultations cost $300 [1]." in body


def test_chat_stream_rejects_invalid_input(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat/stream", json={"query": "x" * 60})

    assert response.status_code == 400


def test_job_status_returns_404_for_unknown_job(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.get("/jobs/does-not-exist")

    assert response.status_code == 404


def test_ingestion_job_lifecycle_records_success(tmp_path: Path, monkeypatch) -> None:
    from api import routes

    state = make_state(tmp_path)
    source = tmp_path / "notes.txt"
    source.write_text("The Chicago office serves Midwest clients.", encoding="utf-8")
    job_id = routes._create_job(state, "ingest")

    monkeypatch.setattr(
        routes,
        "ingest_file",
        lambda *args, **kwargs: [
            Document(page_content="notes", metadata={"source_file": "notes.txt"})
        ],
    )
    monkeypatch.setattr(routes, "_index_documents", lambda *args, **kwargs: 4)

    routes._ingest_path_task(state, job_id, source, source.parent)

    client = client_for(state)
    payload = client.get(f"/jobs/{job_id}").json()

    assert payload["status"] == "succeeded"
    assert payload["documents"] == 1
    assert payload["chunks"] == 4


def test_documents_endpoint_lists_indexed_sources(tmp_path: Path) -> None:
    import json

    state = make_state(tmp_path)
    index_dir = state.settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": "a",
            "page_content": "billing",
            "metadata": {"source_file": "rates.md"},
        },
        {
            "id": "b",
            "page_content": "hours",
            "metadata": {"source_file": "office.md"},
        },
        {
            "id": "c",
            "page_content": "more billing",
            "metadata": {"source_file": "rates.md"},
        },
    ]
    (index_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    client = client_for(state)
    payload = client.get("/documents").json()

    assert payload["total_chunks"] == 3
    assert payload["documents"] == [
        {"source_file": "office.md", "chunks": 1},
        {"source_file": "rates.md", "chunks": 2},
    ]


def test_ui_endpoint_returns_html(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    client = client_for(state)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Law Help" in response.text
    assert "/static/logo.svg" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_static_assets_are_served(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    client = client_for(state)

    script = client.get("/static/app.js")
    stylesheet = client.get("/static/styles.css")
    logo = client.get("/static/logo.svg")

    assert script.status_code == 200
    assert "text/javascript" in script.headers["content-type"]
    assert stylesheet.status_code == 200
    assert "text/css" in stylesheet.headers["content-type"]
    assert logo.status_code == 200
    assert "image/svg+xml" in logo.headers["content-type"]
    assert "/static/app.js" in client.get("/").text


def test_ingest_endpoint_requires_input(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/ingest")

    assert response.status_code == 400


def test_ingest_endpoint_requires_admin_token_in_staging(tmp_path: Path) -> None:
    state = make_state(tmp_path, environment="staging", admin_api_token="correct-token")

    client = client_for(state)
    missing = client.post("/ingest")
    wrong = client.post("/ingest", headers={"X-Admin-Token": "wrong"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json()["detail"]["code"] == "admin_auth_required"


def test_filename_sanitization_prevents_traversal() -> None:
    assert sanitize_filename("..\\..\\evil.txt") == "evil.txt"
    assert sanitize_filename("/tmp/notes.md") == "notes.md"
    assert sanitize_filename("report <final>.pdf") == "report _final_.pdf"


def test_filename_sanitization_rejects_unsupported_types() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        sanitize_filename("../../etc/passwd")
