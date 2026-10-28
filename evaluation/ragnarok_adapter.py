"""Optional ragnarok-ai integration for retrieval metrics and test generation."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from ragnarok_ai.core.types import RAGResponse


class LawFirmRAGPipeline:
    """Adapter from our evaluation pipeline to ragnarok-ai's protocol."""

    def __init__(self, run: Any) -> None:
        self._run = run

    async def query(self, question: str) -> RAGResponse:
        result = self._run(question)
        retrieved_docs = [
            Document(page_content=context, metadata={"id": source_id})
            for context, source_id in zip(
                result.contexts,
                result.retrieved_source_ids or result.citations or result.contexts,
                strict=False,
            )
        ]
        return RAGResponse(answer=result.answer, retrieved_docs=retrieved_docs)


def run_ragnarok_evaluation(pipeline, testset, *, k: int = 10):
    """Run ragnarok-ai evaluation if the optional dependency is installed."""

    from ragnarok_ai import evaluate

    return evaluate(rag_pipeline=pipeline, testset=testset, k=k)
