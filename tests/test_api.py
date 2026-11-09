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


def test_ui_endpoint_returns_html(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    client = client_for(state)
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Law Firm RAG Chatbot" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


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
