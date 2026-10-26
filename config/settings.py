"""Typed, validated application configuration.

Every setting is read from an environment variable with the same name in
upper-case (for example ``embedding_model`` -> ``EMBEDDING_MODEL``). Relative
paths are resolved against the project root, not the current working
directory, so the CLI and the API always agree on where data lives.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.presets import get_profile

PATH_FIELDS: tuple[str, ...] = (
    "data_dir",
    "artifacts_dir",
    "local_model_cache_dir",
    "upload_dir",
    "chroma_persist_dir",
    "qdrant_local_path",
    "golden_set_path",
    "eval_output_dir",
    "ragnarok_checkpoint_dir",
    "log_dir",
    "trace_dir",
)


def discover_project_root() -> Path:
    """Find the repository root by looking for ``pyproject.toml``.

    ``PROJECT_ROOT`` in the process environment takes precedence. This is
    intentionally read before the ``.env`` file because the location of the
    ``.env`` file itself depends on the project root.
    """

    override = os.environ.get("PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    return Path.cwd().resolve()


PROJECT_ROOT = discover_project_root()


class Settings(BaseSettings):
    """Single source of truth for runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_name: str = "Law Firm RAG"
    environment: Literal["local", "staging", "test"] = "local"
    resource_profile: Literal["standard", "cpu"] = "standard"
    project_root: Path = PROJECT_ROOT
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("var/artifacts")
    local_model_cache_dir: Path = Path(".cache/huggingface")
    upload_dir: Path = Path("var/uploads")

    pdf_parser: Literal["pdfmux", "unstructured", "docling"] = "pdfmux"
    chunk_strategy: Literal["recursive", "structure"] = "structure"
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=96, ge=0)
    min_chunk_chars: int = Field(default=50, ge=0, le=10000)
    preserve_tables: bool = True
    deduplicate_by_hash: bool = True
    ingestion_batch_size: int = Field(default=64, ge=1, le=1024)

    embedding_model: str = ""
    embedding_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    embedding_batch_size: int = Field(default=0, ge=0, le=1024)
    embedding_max_seq_length: int = Field(default=0, ge=0, le=131072)
    embedding_normalize: bool = True
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""

    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    vector_store_collection: str = Field(default="lawfirm_docs", min_length=1)
    chroma_persist_dir: Path = Path("var/chroma")
    qdrant_url: str | None = None
    qdrant_local_path: Path = Path("var/qdrant")
    top_k: int = Field(default=6, ge=1, le=50)
    dense_candidates: int = Field(default=20, ge=1, le=200)
    sparse_candidates: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_k1: float = Field(default=1.5, ge=0.0, le=10.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    reranker_enabled: bool = True
    reranker_model: str = ""
    rerank_top_n: int = Field(default=6, ge=1, le=50)
    reranker_batch_size: int = Field(default=0, ge=0, le=1024)
    reranker_max_length: int = Field(default=512, ge=64, le=8192)
    reranker_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    llm_provider: Literal["ollama", "vllm"] = "ollama"
    llm_model: str = ""
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: SecretStr | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(default=8192, ge=512, le=131072)
    llm_max_tokens: int = Field(default=1024, ge=64, le=32768)
    llm_timeout_seconds: int = Field(default=180, ge=10, le=3600)
    llm_keep_alive: str = "5m"

    max_query_rewrites: int = Field(default=2, ge=0, le=5)
    max_grounding_retries: int = Field(default=1, ge=0, le=5)
    max_graph_steps: int = Field(default=12, ge=6, le=50)

    golden_set_path: Path = Path("evaluation/golden_set.jsonl")
    eval_output_dir: Path = Path("var/eval")
    eval_groundedness_min: float = Field(default=0.85, ge=0.0, le=1.0)
    eval_completeness_min: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_relevance_min: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_consistency_min: float = Field(default=0.80, ge=0.0, le=1.0)
    eval_confidence_min: float = Field(default=0.70, ge=0.0, le=1.0)
    eval_iqs_min: float = Field(default=0.80, ge=0.0, le=1.0)
    eval_retrieval_precision_min: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_fail_on_regression: bool = True
    ragas_enabled: bool = False
    eval_judge_model: str = ""
    ragnarok_checkpoint_dir: Path = Path("var/eval/checkpoints")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = True
    log_dir: Path = Path("var/logs")
    trace_dir: Path = Path("var/traces")
    langfuse_enabled: bool = False
    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    max_query_chars: int = Field(default=2000, ge=1, le=100000)
    pii_detection_enabled: bool = False
    rate_limit_per_minute: int = Field(default=60, ge=1, le=100000)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=64)

    @field_validator("llm_base_url")
    @classmethod
    def _validate_llm_base_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LLM_BASE_URL must be an absolute http(s) URL")
        return value.rstrip("/")

    @field_validator("langfuse_host")
    @classmethod
    def _validate_langfuse_host(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LANGFUSE_HOST must be an absolute http(s) URL")
        return value.rstrip("/")

    @field_validator("qdrant_url")
    @classmethod
    def _validate_qdrant_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("QDRANT_URL must be an absolute http(s) URL when set")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _apply_profile_and_validate(self) -> Settings:
        profile = get_profile(self.resource_profile)

        if not self.embedding_model.strip():
            self.embedding_model = profile.embedding_model
        if self.embedding_max_seq_length == 0:
            self.embedding_max_seq_length = profile.embedding_max_seq_length
        if self.embedding_batch_size == 0:
            self.embedding_batch_size = profile.embedding_batch_size
        if not self.reranker_model.strip():
            self.reranker_model = profile.reranker_model
        if self.reranker_batch_size == 0:
            self.reranker_batch_size = profile.reranker_batch_size
        if not self.llm_model.strip():
            self.llm_model = profile.llm_model
        if not self.eval_judge_model.strip():
            self.eval_judge_model = self.llm_model

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.embedding_max_seq_length < self.chunk_size:
            raise ValueError(
                "EMBEDDING_MAX_SEQ_LENGTH must be at least CHUNK_SIZE to avoid silent truncation"
            )
        if self.top_k > self.rerank_top_n:
            raise ValueError("TOP_K cannot exceed RERANK_TOP_N")
        if self.rerank_top_n > self.dense_candidates:
            raise ValueError("RERANK_TOP_N cannot exceed DENSE_CANDIDATES")
        if self.rerank_top_n > self.sparse_candidates:
            raise ValueError("RERANK_TOP_N cannot exceed SPARSE_CANDIDATES")
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one retrieval weight must be greater than zero")
        if self.llm_max_tokens >= self.llm_num_ctx:
            raise ValueError("LLM_MAX_TOKENS must be smaller than LLM_NUM_CTX")
        qdrant_has_target = bool(self.qdrant_url or self.qdrant_local_path)
        if self.vector_store_backend == "qdrant" and not qdrant_has_target:
            raise ValueError("Qdrant requires QDRANT_URL or QDRANT_LOCAL_PATH")

        self.project_root = self.project_root.expanduser().resolve()
        for field_name in PATH_FIELDS:
            raw = getattr(self, field_name)
            setattr(self, field_name, self._resolve_path(raw))

        return self

    def _resolve_path(self, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def ensure_directories(self) -> list[Path]:
        """Create the runtime directories required by the application."""

        paths = [
            self.data_dir,
            self.artifacts_dir,
            self.local_model_cache_dir,
            self.upload_dir,
            self.log_dir,
            self.trace_dir,
            self.eval_output_dir,
            self.ragnarok_checkpoint_dir,
        ]
        if self.vector_store_backend == "chroma":
            paths.append(self.chroma_persist_dir)
        elif self.qdrant_url is None:
            paths.append(self.qdrant_local_path)

        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def public_dict(self) -> dict[str, object]:
        """Return a JSON-safe, secret-redacted representation."""

        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""

    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and reload settings, mainly for tests and tooling."""

    get_settings.cache_clear()
    return get_settings()
