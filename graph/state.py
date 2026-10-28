"""LangGraph state and dependency definitions for the RAG pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from config import Settings


class RAGState(TypedDict, total=False):
    """Typed state carried through the RAG graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    original_query: str
    standalone_query: str
    retrieved_chunks: list[Document]
    retrieved_source_ids: list[str]
    reranked_chunks: list[Document]
    generation: str
    grounding_verdict: str
    citations: list[str]
    route_decision: str
    rewrite_count: int
    grounding_retry_count: int


@dataclass
class GraphDependencies:
    """Runtime dependencies injected into graph nodes."""

    settings: Settings
    llm: BaseChatModel
    retriever: Any
    reranker: Any
    router: Callable[[str], str] | None = None
    condenser: Callable[[str, list[BaseMessage]], str] | None = None
    grader: Callable[[str, list[Document]], list[bool]] | None = None
    rewriter: Callable[[str, list[BaseMessage]], str] | None = None
    generator: Callable[[str, list[Document]], tuple[str, list[str]]] | None = None
    grounding_checker: Callable[[str, list[Document]], bool] | None = None
    direct_answerer: Callable[[str], str] | None = None
    node_latencies: dict[str, float] = field(default_factory=dict)
