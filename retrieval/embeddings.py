"""Local embedding model construction."""

from __future__ import annotations

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from config import apply_model_cache_environment, get_settings


class PrefixedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """Hugging Face embeddings with optional document/query prefixes."""

    query_prefix: str = ""
    passage_prefix: str = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self.passage_prefix}{text}" for text in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{self.query_prefix}{text}")


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to a concrete local device."""

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def create_embeddings(settings=None):
    """Create the configured local embedding model."""

    settings = settings or get_settings()
    apply_model_cache_environment(settings)

    device = resolve_device(settings.embedding_device)
    model_kwargs: dict[str, object] = {"device": device}
    if "nomic" in settings.embedding_model.lower():
        model_kwargs["trust_remote_code"] = True

    encode_kwargs = {
        "normalize_embeddings": settings.embedding_normalize,
        "batch_size": settings.embedding_batch_size,
    }
    query_encode_kwargs = {
        "normalize_embeddings": settings.embedding_normalize,
        "batch_size": 1,
    }

    return PrefixedHuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=str(settings.local_model_cache_dir / "sentence-transformers"),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
    )
