"""Local cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, Field

from config import apply_model_cache_environment, get_settings
from retrieval.embeddings import resolve_device


class CrossEncoderReranker(BaseDocumentCompressor):
    """Rerank documents with a local sentence-transformers cross-encoder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 6
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"
    enabled: bool = True
    model: Any = Field(default=None, exclude=True)

    _loaded_model: Any = None

    def _resolve_device(self) -> str:
        return resolve_device(self.device)

    def _load_model(self):
        if self.model is not None:
            return self.model
        if self._loaded_model is not None:
            return self._loaded_model

        settings = get_settings()
        apply_model_cache_environment(settings)

        from sentence_transformers import CrossEncoder

        device = self._resolve_device()
        loaded = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=device,
        )
        self._loaded_model = loaded
        return loaded

    def compress_documents(self, documents, query, callbacks=None):
        documents = list(documents or [])
        if not documents:
            return []
        if not self.enabled:
            return documents[: self.top_n]

        model = self._load_model()
        pairs = [(query, document.page_content) for document in documents]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(
            zip(documents, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        result: list[Document] = []
        for document, score in ranked[: self.top_n]:
            metadata = dict(document.metadata or {})
            metadata["reranker_score"] = float(score)
            result.append(Document(page_content=document.page_content, metadata=metadata))

        return result
