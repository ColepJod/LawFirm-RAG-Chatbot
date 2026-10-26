"""Model presets used to keep local, CPU-only deployments practical.

The profile only supplies defaults. An explicit environment variable such as
``EMBEDDING_MODEL`` always wins, which prevents accidental model changes after
a vector index has already been built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResourceProfile = Literal["standard", "cpu"]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A coherent set of local models for one hardware profile."""

    embedding_model: str
    embedding_max_seq_length: int
    embedding_batch_size: int
    reranker_model: str
    reranker_max_length: int
    reranker_batch_size: int
    llm_model: str


RESOURCE_PROFILES: dict[ResourceProfile, ModelProfile] = {
    "standard": ModelProfile(
        embedding_model="BAAI/bge-m3",
        embedding_max_seq_length=8192,
        embedding_batch_size=32,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_max_length=512,
        reranker_batch_size=16,
        llm_model="qwen3:8b",
    ),
    "cpu": ModelProfile(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_max_seq_length=512,
        embedding_batch_size=16,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_max_length=512,
        reranker_batch_size=8,
        llm_model="qwen3:4b",
    ),
}


def get_profile(name: ResourceProfile) -> ModelProfile:
    """Return the profile for ``name``.

    ``name`` is already constrained by the settings Literal, so a missing key
    indicates an internal inconsistency rather than bad user input.
    """

    return RESOURCE_PROFILES[name]
