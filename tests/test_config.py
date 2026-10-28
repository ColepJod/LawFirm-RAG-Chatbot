from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import PROJECT_ROOT, Settings, apply_model_cache_environment, get_settings


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_standard_profile_defaults() -> None:
    settings = make_settings()

    assert settings.resource_profile == "standard"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.eval_judge_model == "llama-3.3-70b-versatile"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 96
    assert 0.10 <= settings.chunk_overlap / settings.chunk_size <= 0.25
    assert settings.top_k == 6
    assert settings.rrf_k == 60
    assert settings.vector_store_backend == "chroma"
    assert settings.eval_groundedness_min == 0.85
    assert settings.eval_iqs_min == 0.80


def test_cpu_profile_defaults() -> None:
    settings = make_settings(resource_profile="cpu")

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_max_seq_length == 512
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.eval_judge_model == "llama-3.3-70b-versatile"


def test_explicit_model_overrides_profile() -> None:
    settings = make_settings(
        resource_profile="cpu",
        embedding_model="custom/local-embedding",
        reranker_model="custom/local-reranker",
        llm_model="custom:7b",
        eval_judge_model="judge:2b",
    )

    assert settings.embedding_model == "custom/local-embedding"
    assert settings.reranker_model == "custom/local-reranker"
    assert settings.llm_model == "custom:7b"
    assert settings.eval_judge_model == "judge:2b"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESOURCE_PROFILE", "cpu")
    monkeypatch.setenv("EMBEDDING_MODEL", "env/model")
    monkeypatch.setenv("TOP_K", "4")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "qdrant")

    settings = make_settings()

    assert settings.embedding_model == "env/model"
    assert settings.top_k == 4
    assert settings.vector_store_backend == "qdrant"


def test_empty_environment_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_API_KEY", "")

    settings = make_settings(resource_profile="cpu")

    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.llm_api_key is None


def test_invalid_vector_store_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(vector_store_backend="faiss")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_overlap=512)


def test_top_k_cannot_exceed_rerank_top_n() -> None:
    with pytest.raises(ValidationError, match="TOP_K"):
        make_settings(top_k=7)


def test_rerank_top_n_cannot_exceed_candidate_depth() -> None:
    with pytest.raises(ValidationError, match="DENSE_CANDIDATES"):
        make_settings(rerank_top_n=21)


def test_retrieval_weights_cannot_both_be_zero() -> None:
    with pytest.raises(ValidationError, match="retrieval weight"):
        make_settings(dense_weight=0.0, sparse_weight=0.0)


def test_embedding_context_must_cover_chunk_size() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_MAX_SEQ_LENGTH"):
        make_settings(embedding_max_seq_length=256)


def test_llm_output_must_fit_inside_context() -> None:
    with pytest.raises(ValidationError, match="LLM_MAX_TOKENS"):
        make_settings(llm_max_tokens=8192, llm_num_ctx=8192)


def test_invalid_qdrant_url_is_rejected() -> None:
    with pytest.raises(ValidationError, match="QDRANT_URL"):
        make_settings(qdrant_url="localhost:6333")


def test_relative_paths_resolve_against_project_root() -> None:
    settings = make_settings()

    assert settings.project_root == PROJECT_ROOT
    assert settings.chroma_persist_dir == PROJECT_ROOT / "var/chroma"
    assert settings.local_model_cache_dir == PROJECT_ROOT / ".cache/huggingface"


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute-cache"
    settings = make_settings(local_model_cache_dir=absolute)

    assert settings.local_model_cache_dir == absolute.resolve()


def test_ensure_directories_is_idempotent(tmp_path: Path) -> None:
    settings = make_settings(project_root=tmp_path)

    first = settings.ensure_directories()
    second = settings.ensure_directories()

    assert first == second
    assert settings.data_dir.is_dir()
    assert settings.local_model_cache_dir.is_dir()
    assert settings.chroma_persist_dir.is_dir()
    assert settings.log_dir.is_dir()
    assert settings.trace_dir.is_dir()


def test_secrets_are_redacted_in_public_dump() -> None:
    settings = make_settings(
        llm_api_key="super-secret",
        langfuse_public_key="public-key",
        langfuse_secret_key="private-key",
    )

    dumped = json.dumps(settings.public_dict())

    assert "super-secret" not in dumped
    assert "private-key" not in dumped
    assert "**********" in dumped


def test_model_cache_environment_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "hf-cache"
    settings = make_settings(local_model_cache_dir=cache_dir)

    effective = apply_model_cache_environment(settings)

    assert cache_dir.is_dir()
    assert effective["HF_HOME"] == str(cache_dir)
    assert effective["SENTENCE_TRANSFORMERS_HOME"] == str(cache_dir / "sentence-transformers")
    assert effective["TOKENIZERS_PARALLELISM"] == "false"


def test_existing_cache_environment_is_respected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "explicit-hf-home"))
    settings = make_settings(local_model_cache_dir=tmp_path / "configured-cache")

    effective = apply_model_cache_environment(settings)

    assert effective["HF_HOME"] == str(tmp_path / "explicit-hf-home")


def test_get_settings_is_cached() -> None:
    first = get_settings()
    second = get_settings()

    assert first is second


def test_env_example_is_loadable() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")

    assert settings.resource_profile == "cpu"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.llm_api_key is None


def test_env_example_documents_every_setting() -> None:
    example = PROJECT_ROOT / ".env.example"
    documented: set[str] = set()

    for raw_line in example.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        line = line.lstrip("#").strip()
        name = line.split("=", 1)[0].strip()
        if name == name.upper() and name.replace("_", "").isalnum():
            documented.add(name)

    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= documented, f"Missing from .env.example: {sorted(expected - documented)}"
