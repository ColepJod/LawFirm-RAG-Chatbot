"""Node implementations for the LangGraph RAG control loop."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage

from chunking.tokens import count_tokens, get_encoding
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


def _invoke_json_bool(llm: Any, prompt: str, key: str) -> bool:
    """Invoke the LLM with a JSON boolean answer, falling back to a yes/no answer.

    Some instruction-tuned chat models ignore "answer only yes or no" style
    prompts and return unreliable free-form text. Structured JSON answers are
    substantially more deterministic, with a tolerant fallback for plain
    "yes"/"no" responses (and test doubles).
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    text = str(getattr(response, "content", response) or "").strip()
    normalized = text.strip().strip("`").strip()
    if normalized.lower().startswith("json"):
        normalized = normalized[4:].strip().lstrip("`\n ").rstrip("`\n ")
    parsed: Any = None
    try:
        parsed = json.loads(normalized)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None

    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, dict) and key in parsed:
        value = parsed[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "true", "y", "1"}:
                return True
            if lowered in {"no", "false", "n", "0"}:
                return False
    return text.lower().startswith("yes")


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


def _format_context_block(index: int, document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source_file") or metadata.get("source") or "unknown"
    section = metadata.get("section_heading") or metadata.get("title") or "document"
    page = metadata.get("page_number") or metadata.get("page_start")
    page_text = f"p. {page}" if page is not None else "p. n/a"
    return (
        f"[{index}] Source: {source} | Section: {section} | Page: {page_text}\n"
        f"{document.page_content}"
    )


def _source_payload(index: int, document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    page = metadata.get("page_number") or metadata.get("page_start")
    try:
        page_number = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_number = None
    score = metadata.get("reranker_score", metadata.get("rrf_score", metadata.get("bm25_score")))
    try:
        score_value = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_value = None
    snippet = " ".join((document.page_content or "").split())
    return {
        "id": index,
        "source_file": str(metadata.get("source_file") or metadata.get("source") or "unknown"),
        "section": str(metadata.get("section_heading") or metadata.get("title") or "document"),
        "page": page_number,
        "chunk_id": _source_id(document),
        "score": score_value,
        "snippet": snippet[:420],
    }


def _source_id(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("chunk_id", "id", "document_hash"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return f"content:{hashlib.sha256((document.page_content or '').encode('utf-8')).hexdigest()}"


def _extract_citation_indices(answer: str, document_count: int) -> list[int]:
    """Extract leading numeric labels from square-bracket citations.

    Handles ``[1]``, ``[1, 2]``, and ``[1, Section, p. 3]`` without treating a
    page number later in the bracket as a separate source label.
    """

    indices: set[int] = set()
    for content in re.findall(r"\[([^\]]+)\]", answer or ""):
        for token in re.split(r"[,\s]+", content.strip()):
            if not token:
                continue
            if token.isdigit():
                value = int(token)
                if 1 <= value <= document_count:
                    indices.add(value)
            else:
                break
    return sorted(indices)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    encoding = get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def _select_context(documents: list[Document], max_tokens: int) -> list[Document]:
    """Select documents that fit the generation context budget."""

    selected: list[Document] = []
    used = 0
    for document in documents:
        block = _format_context_block(len(selected) + 1, document)
        block_tokens = count_tokens(block)
        if used + block_tokens <= max_tokens:
            selected.append(document)
            used += block_tokens
            continue
        if not selected:
            # Keep at least one source, truncating its body rather than dropping it.
            header_tokens = count_tokens(
                _format_context_block(1, Document(page_content="", metadata=document.metadata))
            )
            budget = max(1, max_tokens - header_tokens)
            truncated = _truncate_to_tokens(document.page_content, budget)
            selected.append(
                Document(page_content=truncated, metadata=dict(document.metadata or {}))
            )
        break
    return selected


def _heuristic_route(query: str) -> str | None:
    normalized = (query or "").strip().lower()
    if not normalized:
        return "chit_chat"

    firm_terms = (
        "firm",
        "attorney",
        "lawyer",
        "rate",
        "fee",
        "billing",
        "office",
        "location",
        "practice",
        "specialt",
        "matter",
        "client",
        "intake",
        "policy",
        "consultation",
        "contact",
        "phone",
        "email",
        "cost",
        "price",
        "confidential",
        "security",
        "insurance",
        "liability",
        "patent",
        "litigation",
        "employment",
        "corporate",
        "real estate",
        "intellectual property",
        "pro bono",
        "new york",
        "chicago",
        "california",
        "washington",
        "manhattan",
        "document",
        "contract",
        "compliance",
    )
    if any(term in normalized for term in firm_terms):
        return "rag"

    if (
        re.match(r"^(hi|hello|hey|good morning|good afternoon|thanks|thank you)\b", normalized)
        and len(normalized.split()) <= 7
    ):
        return "chit_chat"

    return None


def default_route(llm: Any, query: str) -> str:
    heuristic = _heuristic_route(query)
    if heuristic is not None:
        return heuristic

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
        response = _invoke_json_bool(
            llm,
            GRADE_PROMPT.format(query=query, document=document.page_content[:1500]),
            "relevant",
        )
        results.append(bool(response))
    return results


def default_generate(
    llm: Any,
    query: str,
    documents: list[Document],
    *,
    max_context_tokens: int = 4096,
) -> tuple[str, list[str]]:
    if not documents:
        return (
            "I do not have enough information in the retrieved documents to answer that question.",
            [],
        )
    selected = _select_context(documents, max_context_tokens)
    context = "\n\n".join(
        _format_context_block(index + 1, document) for index, document in enumerate(selected)
    )
    answer = _invoke_text(llm, GENERATE_PROMPT.format(query=query, context=context))
    citations = [_format_citation(document) for document in selected]
    return answer, citations


def default_grounding(
    llm: Any,
    answer: str,
    documents: list[Document],
    *,
    max_context_tokens: int = 4096,
) -> bool:
    if not documents:
        return True
    if "do not have enough information" in answer.lower():
        return True
    selected = _select_context(documents, max_context_tokens)
    context = "\n\n".join(
        _format_context_block(index + 1, document) for index, document in enumerate(selected)
    )
    return bool(
        _invoke_json_bool(
            llm,
            GROUNDING_PROMPT.format(context=context, answer=answer),
            "grounded",
        )
    )


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
            "retrieved_source_ids": [],
            "reranked_chunks": [],
            "citations": [],
            "sources": [],
            "citation_warnings": [],
            "generation": "",
            "grounding_verdict": "",
            "node_latencies": {},
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
        return {
            "retrieved_chunks": list(documents),
            "retrieved_source_ids": [_source_id(document) for document in documents],
        }

    return retrieve


def make_grade_node(dependencies: GraphDependencies):
    def grade(state: RAGState) -> dict[str, Any]:
        query = (
            state.get("standalone_query") or state.get("original_query") or _current_query(state)
        )
        documents = list(state.get("retrieved_chunks", []))

        if dependencies.grader is not None:
            flags = dependencies.grader(query, documents)
            relevant = [
                document
                for document, is_relevant in zip(documents, flags, strict=False)
                if is_relevant
            ]
        elif dependencies.settings.grader_mode == "llm":
            flags = default_grade(dependencies.llm, query, documents)
            relevant = [
                document
                for document, is_relevant in zip(documents, flags, strict=False)
                if is_relevant
            ]
        else:
            # Default: the local cross-encoder in the next node decides relevance.
            # This removes one LLM call per retrieved document from the hot path.
            relevant = documents

        # Safety net for LLM grading: when the judge finds nothing relevant, fall
        # back to the deterministic reranker before triggering a corrective
        # rewrite. LLM judges can produce false negatives, which would otherwise
        # discard good candidates.
        reranker = dependencies.reranker
        reranker_usable = reranker is not None and bool(getattr(reranker, "enabled", True))
        if (
            dependencies.settings.grader_mode == "llm"
            and not relevant
            and documents
            and reranker_usable
        ):
            reranked = list(reranker.compress_documents(list(documents), query))
            kept_contents = {document.page_content for document in reranked}
            relevant = [
                document for document in documents if document.page_content in kept_contents
            ]

        return {
            "retrieved_chunks": relevant,
            "retrieved_source_ids": [_source_id(document) for document in relevant],
        }

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
        threshold = dependencies.settings.retrieval_min_rerank_score
        filtered = []
        for document in reranked:
            score = (document.metadata or {}).get("reranker_score")
            try:
                if score is not None and float(score) < threshold:
                    continue
            except (TypeError, ValueError):
                pass
            filtered.append(document)
        return {
            "reranked_chunks": filtered,
            "retrieved_source_ids": [_source_id(document) for document in filtered]
            or state.get("retrieved_source_ids", []),
        }

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
        selected = _select_context(documents, dependencies.settings.context_max_tokens)
        generator = dependencies.generator
        if generator is None:
            answer, citations = default_generate(
                dependencies.llm,
                query,
                selected,
                max_context_tokens=dependencies.settings.context_max_tokens,
            )
        else:
            answer, citations = generator(query, selected)

        cited_indices = _extract_citation_indices(answer, len(selected))
        if cited_indices:
            sources = [_source_payload(index, selected[index - 1]) for index in cited_indices]
            warnings: list[str] = []
        else:
            sources = [
                _source_payload(index + 1, document) for index, document in enumerate(selected)
            ]
            warnings = (
                ["The answer did not contain a citation marker; sources are shown unranked."]
                if selected
                else []
            )
        return {
            "generation": answer,
            "citations": citations,
            "sources": sources,
            "citation_warnings": warnings,
        }

    return generate


def make_grounding_node(dependencies: GraphDependencies):
    def check(state: RAGState) -> dict[str, Any]:
        answer = str(state.get("generation", ""))
        documents = list(state.get("reranked_chunks") or state.get("retrieved_chunks", []))
        checker = dependencies.grounding_checker or (
            lambda value, docs: default_grounding(
                dependencies.llm,
                value,
                docs,
                max_context_tokens=dependencies.settings.context_max_tokens,
            )
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
            "sources": [],
            "citation_warnings": [],
            "grounding_verdict": "pass",
        }

    return direct_answer


def make_log_trace_node(dependencies: GraphDependencies):
    from observability.tracing import RunTrace, TraceLogger

    logger = dependencies.trace_logger
    if logger is None:
        from config import get_settings

        logger = TraceLogger(dependencies.settings or get_settings())
        dependencies.trace_logger = logger

    def log_trace(state: RAGState) -> dict[str, Any]:
        reranked = list(state.get("reranked_chunks") or [])
        retrieved = list(state.get("retrieved_chunks", []))
        documents = reranked or retrieved
        trace = RunTrace(
            query=str(state.get("original_query", "")),
            standalone_query=str(state.get("standalone_query", "")),
            retrieved_chunk_ids=[
                str((document.metadata or {}).get("chunk_id") or _source_id(document))
                for document in retrieved
            ],
            reranker_scores=[
                float((document.metadata or {}).get("reranker_score", 0.0)) for document in reranked
            ],
            route_decision=str(state.get("route_decision", "")),
            grounding_verdict=str(state.get("grounding_verdict", "")),
            final_context_length=sum(len(document.page_content) for document in documents),
            response=str(state.get("generation", "")),
            node_latencies_ms=dict(state.get("node_latencies", {})),
        )
        logger.write(trace)
        return {}

    return log_trace
