from __future__ import annotations

import json
from pathlib import Path

from config import Settings
from observability import InputGuardrails, OutputGuardrails, RunTrace, TraceLogger, health_check


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, project_root=tmp_path, max_query_chars=5)


def test_input_guardrail_rejects_long_query(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = InputGuardrails(settings).validate("too long")

    assert result.allowed is False
    assert any("exceeds" in reason for reason in result.reasons)


def test_input_guardrail_accepts_short_query(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = InputGuardrails(settings).validate("ok")

    assert result.allowed is True


def test_output_guardrail_requires_source_citation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    doc = type("Doc", (), {"metadata": {"source_file": "rates.md"}})()
    result = OutputGuardrails(settings).validate_citations("No citation here.", [doc])

    assert result.allowed is False


def test_output_guardrail_accepts_numbered_citation_answer(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    doc = type("Doc", (), {"metadata": {"source_file": "rag_chatbot.txt"}})()
    answer = (
        "Hawthorne Bell & Lawson LLP (HBL) is a fictional law firm "
        "[2, Document Purpose and Entity Identification, p. 1]."
    )
    result = OutputGuardrails(settings).validate_citations(answer, [doc])

    assert result.allowed is True


def test_output_guardrail_accepts_refusal_without_context(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = OutputGuardrails(settings).validate_refusal("I do not have enough information.", [])

    assert result.allowed is True


def test_output_guardrail_allows_non_context_answer_for_chit_chat(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    result = OutputGuardrails(settings).validate_refusal("Hello! How can I assist you today?", [])

    assert result.allowed is True
    assert result.reasons == []


def test_health_check_reports_checks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    report = health_check(settings)

    assert "status" in report
    assert "checks" in report
    assert "model_cache" in report["checks"]


def test_trace_logger_writes_json_line(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    logger = TraceLogger(settings, path=tmp_path / "traces.jsonl")
    trace = RunTrace(query="q", standalone_query="sq", route_decision="rag", response="a")

    logger.write(trace)

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["query"] == "q"
    assert payload["standalone_query"] == "sq"
    assert "ended_at" in payload
