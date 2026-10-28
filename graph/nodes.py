"""Node implementations for the LangGraph RAG control loop."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage

from graph.prompts import (
    CONDENSE_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    GROUNDING_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
)
from graph.state import GraphDependencies, RAGState


def _invoke_text(llm: Any, prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", response)
    return str(content).strip()


def _current_query(state: RAGState) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return str(message.content)
    return str(state.get("original_query", "") or state.get("standalone_query", ""))


def _history(state: RAGState) -> list[BaseMessage]:
    messages = list(state.get("messages", []))
    return messages[:-1] if messages else []


def _retrieve(retriever: Any, query: str) -> list[Document]:
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query))
    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(query))
    raise TypeError("retriever must implement invoke or get_relevant_documents")


def _rerank(reranker: Any, query: str, documents: list[Document]) -> list[Document]:
    if not documents:
        return []
    if hasattr(reranker, "compress_documents"):
        return list(reranker.compress_documents(documents, query))
    if callable(reranker):
        return list(reranker(documents, query))
    raise TypeError("reranker must implement compress_documents or be callable")


def _format_citation(document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source_file") or metadata.get("source") or "unknown"
    section = metadata.get("section_heading") or metadata.get("title") or "document"
    page = metadata.get("page_number") or metadata.get("page_start")
    page_text = f"p. {page}" if page is not None else "p. n/a"
    return f"{source} | {section} | {page_text}"


def default_route(llm: Any, query: str) -> str:
    response = _invoke_text(llm, ROUTE_PROMPT.format(query=query)).lower()
    if "chit" in response:
        return "chit_chat"
    if "out" in response:
        return "out_of_scope"
    return "rag"


def default_condense(llm: Any, query: str, history: list[BaseMessage]) -> str:
    if not history:
        return query
    history_text = "\n".join(f"{message.type}: {message.content}" for message in history[-6:])
    return _invoke_text(llm, CONDENSE_PROMPT.format(history=history_text, query=query))


def default_rewrite(llm: Any, query: str, history: list[BaseMessage]) -> str:
    return _invoke_text(llm, REWRITE_PROMPT.format(query=query))


def default_grade(llm: Any, query: str, documents: list[Document]) -> list[bool]:
    results: list[bool] = []
    for document in documents:
        response = _invoke_text(
            llm,
            GRADE_PROMPT.format(query=query, document=document.page_content[:1500]),
        ).lower()
        results.append(response.startswith("yes"))
    return results


def default_generate(llm: Any, query: str, documents: list[Document]) -> tuple[str, list[str]]:
    if not documents:
        return (
            "I do not have enough information in the retrieved documents to answer that question.",
            [],
        )
    context = "\n\n".join(
        f"[{index + 1}] {document.page_content}" for index, document in enumerate(documents)
    )
    answer = _invoke_text(llm, GENERATE_PROMPT.format(query=query, context=context))
    citations = [_format_citation(document) for document in documents]
    return answer, citations


def default_grounding(llm: Any, answer: str, documents: list[Document]) -> bool:
    if not documents:
        return True
    if "do not have enough information" in answer.lower():
        return True
    context = "\n\n".join(document.page_content for document in documents)
    response = _invoke_text(llm, GROUNDING_PROMPT.format(context=context, answer=answer)).lower()
    return response.startswith("yes")


def default_direct_answer(llm: Any, query: str) -> str:
    return _invoke_text(llm, DIRECT_ANSWER_PROMPT.format(query=query))


def make_route_node(dependencies: GraphDependencies):
    def route(state: RAGState) -> dict[str, Any]:
        query = _current_query(state)
        router = dependencies.router or (lambda value: default_route(dependencies.llm, value))
        decision = router(query)
        if decision not in {"rag", "chit_chat", "out_of_scope"}:
            decision = "rag"
        return {
            "original_query": query,
            "standalone_query": query,
            "route_decision": decision,
            "rewrite_count": 0,
            "grounding_retry_count": 0,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "citations": [],
            "generation": "",
            "grounding_verdict": "",
        }

    return route


def make_condense_node(dependencies: GraphDependencies):
    def condense(state: RAGState) -> dict[str, Any]:
        query = state.get("original_query") or _current_query(state)
        history = _history(state)
        condenser = dependencies.condenser or (
            lambda value, messages: default_condense(dependencies.llm, value, messages)
        )
        standalone = condenser(query, history)
        return {"standalone_query": standalone or query}

    return condense


def make_retrieve_node(dependencies: GraphDependencies):
    def retrieve(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        documents = _retrieve(dependencies.retriever, query)
        return {"retrieved_chunks": list(documents)}

    return retrieve


def make_grade_node(dependencies: GraphDependencies):
    def grade(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        documents = list(state.get("retrieved_chunks", []))
        grader = dependencies.grader or (
            lambda value, docs: default_grade(dependencies.llm, value, docs)
        )
        flags = grader(query, documents)
        relevant = [
            document for document, is_relevant in zip(documents, flags, strict=False) if is_relevant
        ]
        return {"retrieved_chunks": relevant}

    return grade


def make_rerank_node(dependencies: GraphDependencies):
    def rerank(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        documents = list(state.get("retrieved_chunks", []))
        if not documents:
            return {"reranked_chunks": []}
        reranked = _rerank(dependencies.reranker, query, documents)
        return {"reranked_chunks": list(reranked)}

    return rerank


def make_rewrite_node(dependencies: GraphDependencies):
    def rewrite(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        history = _history(state)
        rewriter = dependencies.rewriter or (
            lambda value, messages: default_rewrite(dependencies.llm, value, messages)
        )
        rewritten = rewriter(query, history)
        return {
            "standalone_query": rewritten or query,
            "rewrite_count": int(state.get("rewrite_count", 0)) + 1,
        }

    return rewrite


def make_generate_node(dependencies: GraphDependencies):
    def generate(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        documents = list(state.get("reranked_chunks") or state.get("retrieved_chunks", []))
        generator = dependencies.generator or (
            lambda value, docs: default_generate(dependencies.llm, value, docs)
        )
        answer, citations = generator(query, documents)
        return {"generation": answer, "citations": citations}

    return generate


def make_grounding_node(dependencies: GraphDependencies):
    def check(state: RAGState) -> dict[str, Any]:
        answer = str(state.get("generation", ""))
        documents = list(state.get("reranked_chunks") or state.get("retrieved_chunks", []))
        checker = dependencies.grounding_checker or (
            lambda value, docs: default_grounding(dependencies.llm, value, docs)
        )
        is_grounded = checker(answer, documents)
        if is_grounded:
            return {"grounding_verdict": "pass"}

        retry_count = int(state.get("grounding_retry_count", 0))
        if retry_count < dependencies.settings.max_grounding_retries:
            return {"grounding_verdict": "retry", "grounding_retry_count": retry_count + 1}
        return {"grounding_verdict": "fail"}

    return check


def make_direct_answer_node(dependencies: GraphDependencies):
    def direct_answer(state: RAGState) -> dict[str, Any]:
        query = state.get("original_query") or _current_query(state)
        answerer = dependencies.direct_answerer or (
            lambda value: default_direct_answer(dependencies.llm, value)
        )
        answer = answerer(query)
        return {
            "generation": answer,
            "citations": [],
            "grounding_verdict": "pass",
        }

    return direct_answer
