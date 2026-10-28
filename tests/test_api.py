from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.main import app
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


def make_state(tmp_path: Path) -> AppState:
    settings = Settings(_env_file=None, project_root=tmp_path, max_query_chars=5)
    from observability import InputGuardrails, OutputGuardrails, TraceLogger

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("", encoding="utf-8")
    return AppState(
        settings=settings,
        graph=FakeGraph(
            {
                "generation": "Consultations cost $300 (rates.md).",
                "citations": ["rates.md | Billing | p. 1"],
                "retrieved_source_ids": ["chunk-1"],
                "grounding_verdict": "pass",
                "reranked_chunks": [
                    Document(
                        page_content="Consultations cost $300.",
                        metadata={"source_file": "rates.md"},
                    )
                ],
                "retrieved_chunks": [
                    Document(
                        page_content="Consultations cost $300.",
                        metadata={"source_file": "rates.md"},
                    )
                ],
            }
        ),
        dependencies=None,
        vector_store=FakeVectorStore(),
        bm25_index=FakeBM25(),
        trace_logger=TraceLogger(settings, path=trace_path),
        input_guardrails=InputGuardrails(settings),
        output_guardrails=OutputGuardrails(settings),
        trace_path=trace_path,
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
    assert "checks" in response.json()


def test_metrics_endpoint(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["trace_count"] == 0


def test_chat_endpoint_returns_answer(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": "ok"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Consultations cost $300 (rates.md)."
    assert response.json()["thread_id"] == "default"


def test_chat_endpoint_rejects_long_query(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": "too long"})

    assert response.status_code == 400


def test_chat_endpoint_rejects_empty_query(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/chat", json={"query": ""})

    assert response.status_code == 422


def test_ingest_endpoint_requires_input(tmp_path: Path) -> None:
    state = make_state(tmp_path)

    client = client_for(state)
    response = client.post("/ingest")

    assert response.status_code == 400
