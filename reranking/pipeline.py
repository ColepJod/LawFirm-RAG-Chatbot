"""Reranking entry points used by retrieval and graph nodes."""

from __future__ import annotations

from config import get_settings
from reranking.cross_encoder import CrossEncoderReranker


def create_reranker(settings=None):
    settings = settings or get_settings()
    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        top_n=settings.rerank_top_n,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        device=settings.reranker_device,
        enabled=settings.reranker_enabled,
    )


def rerank_documents(documents, query, *, settings=None, reranker=None):
    reranker = reranker or create_reranker(settings)
    return list(reranker.compress_documents(documents, query))
