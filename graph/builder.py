"""Compile the LangGraph RAG state graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.checkpointing import create_checkpointer
from graph.nodes import (
    make_condense_node,
    make_direct_answer_node,
    make_generate_node,
    make_grade_node,
    make_grounding_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_node,
    make_route_node,
)
from graph.state import GraphDependencies, RAGState


def build_graph(dependencies: GraphDependencies, checkpointer=None):
    """Build and compile the RAG control loop."""

    if checkpointer is None:
        checkpointer = create_checkpointer(dependencies.settings)

    graph = StateGraph(RAGState)

    graph.add_node("route", make_route_node(dependencies))
    graph.add_node("condense_question", make_condense_node(dependencies))
    graph.add_node("retrieve", make_retrieve_node(dependencies))
    graph.add_node("grade_documents", make_grade_node(dependencies))
    graph.add_node("rerank", make_rerank_node(dependencies))
    graph.add_node("rewrite_query", make_rewrite_node(dependencies))
    graph.add_node("generate", make_generate_node(dependencies))
    graph.add_node("check_grounding", make_grounding_node(dependencies))
    graph.add_node("direct_answer", make_direct_answer_node(dependencies))

    graph.add_edge(START, "route")

    def route_branch(state: RAGState) -> str:
        decision = state.get("route_decision", "rag")
        if decision in {"chit_chat", "out_of_scope"}:
            return "direct_answer"
        return "condense_question"

    graph.add_conditional_edges(
        "route",
        route_branch,
        {
            "direct_answer": "direct_answer",
            "condense_question": "condense_question",
        },
    )

    graph.add_edge("condense_question", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    def grade_branch(state: RAGState) -> str:
        documents = state.get("retrieved_chunks", [])
        rewrite_count = int(state.get("rewrite_count", 0))
        max_rewrites = dependencies.settings.max_query_rewrites
        if documents:
            return "rerank"
        if rewrite_count < max_rewrites:
            return "rewrite_query"
        return "generate"

    graph.add_conditional_edges(
        "grade_documents",
        grade_branch,
        {
            "rerank": "rerank",
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    graph.add_edge("rerank", "generate")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check_grounding")

    def grounding_branch(state: RAGState) -> str:
        if state.get("grounding_verdict") == "retry":
            return "generate"
        return END

    graph.add_conditional_edges(
        "check_grounding",
        grounding_branch,
        {
            "generate": "generate",
            END: END,
        },
    )

    graph.add_edge("direct_answer", END)

    return graph.compile(checkpointer=checkpointer)
