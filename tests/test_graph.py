from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from config import Settings
from graph import GraphDependencies, build_graph


class FakeLLM(BaseChatModel):
    _responses: PrivateAttr[list[str]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        text = self._responses.pop(0) if self._responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class FakeRetriever:
    def __init__(self, responses: list[list[Document]]) -> None:
        self.responses = list(responses)
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return list(self.responses.pop(0)) if self.responses else []


class FakeReranker:
    enabled: bool = True

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def compress_documents(self, documents, query):
        return list(documents)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def invoke_graph(
    settings: Settings,
    llm: FakeLLM,
    retriever: FakeRetriever,
    *,
    query: str = "test query",
    **deps_overrides,
) -> dict:
    deps = GraphDependencies(
        settings=settings,
        llm=llm,
        retriever=retriever,
        reranker=FakeReranker(),
        **deps_overrides,
    )
    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 30}
    return graph.invoke({"messages": [HumanMessage(query)]}, config=config)


def test_route_sends_chit_chat_to_direct_answer() -> None:
    settings = make_settings()
    llm = FakeLLM(["chit_chat", "Hello there!"])
    retriever = FakeRetriever([])

    result = invoke_graph(settings, llm, retriever)

    assert result["route_decision"] == "chit_chat"
    assert result["generation"] == "Hello there!"
    assert result["citations"] == []
    assert retriever.queries == []


def test_knowledge_question_routes_to_retrieval() -> None:
    settings = make_settings()
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    llm = FakeLLM(["rag", "Consultations cost $300.", "yes"])
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever)

    assert result["route_decision"] == "rag"
    assert retriever.queries == ["test query"]
    assert result["generation"] == "Consultations cost $300."
    assert result["grounding_verdict"] == "pass"
    assert result["citations"]


def test_correction_rewrites_and_retrieves_again() -> None:
    settings = make_settings(grader_mode="llm")
    irrelevant = Document(
        page_content="Unrelated text.",
        metadata={"source_file": "other.md", "section_heading": "Other", "page_number": 1},
    )
    relevant = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    llm = FakeLLM(
        ["rag", "no", "better billing question", "yes", "Consultations cost $300.", "yes"]
    )
    retriever = FakeRetriever([[irrelevant], [relevant]])
    # Disable the deterministic reranker so the LLM judge alone drives the rewrite.
    deps = GraphDependencies(
        settings=settings,
        llm=llm,
        retriever=retriever,
        reranker=FakeReranker(enabled=False),
    )
    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 30}
    result = graph.invoke({"messages": [HumanMessage("test query")]}, config=config)

    assert retriever.queries == ["test query", "better billing question"]
    assert result["generation"] == "Consultations cost $300."
    assert result["rewrite_count"] == 1


def test_llm_irrelevant_falls_back_to_reranker_before_rewrite() -> None:
    settings = make_settings(grader_mode="llm")
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    llm = FakeLLM(["rag", "no", "Consultations cost $300.", "yes"])
    retriever = FakeRetriever([[doc]])
    deps = GraphDependencies(
        settings=settings,
        llm=llm,
        retriever=retriever,
        reranker=FakeReranker(),
    )
    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 30}
    result = graph.invoke({"messages": [HumanMessage("test query")]}, config=config)

    # The LLM judge flagged the only retrieved document as irrelevant, but the
    # deterministic reranker retained it, so the system should proceed to generate
    # instead of rewriting the query.
    assert retriever.queries == ["test query"]
    assert result["retrieved_chunks"]
    assert result["generation"] == "Consultations cost $300."
    assert result["rewrite_count"] == 0


def test_empty_retrieval_refuses_without_rewrite_when_disabled() -> None:
    settings = make_settings(max_query_rewrites=0)
    llm = FakeLLM(["rag"])
    retriever = FakeRetriever([[]])

    result = invoke_graph(settings, llm, retriever)

    assert result["retrieved_chunks"] == []
    assert "couldn't find that in the indexed documents" in result["generation"].lower()
    assert result["sources"] == []
    assert result["grounding_verdict"] == "pass"


def test_subjective_attorney_question_expands_retrieval_terms() -> None:
    settings = make_settings()
    doc = Document(
        page_content=(
            "Elena M. Vasquez is a partner with twenty years of experience and leads "
            "the Corporate Law and Mergers and Acquisitions practice."
        ),
        metadata={
            "source_file": "directory.md",
            "section_heading": "3.3 Elena M. Vasquez",
            "page_number": 1,
        },
    )
    # "attorney" triggers the deterministic rag route, so only generation and
    # grounding consume LLM responses.
    llm = FakeLLM(
        [
            (
                "The firm's documents don't rank attorneys as \"best.\" They do identify "
                "practice-group leaders and their years of experience [1]."
            ),
            "yes",
        ]
    )
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(
        settings,
        llm,
        retriever,
        query="Which are the firm's best attorneys?",
    )

    assert retriever.queries == [
        "Which are the firm's best attorneys? attorney directory practice group leaders "
        "years in practice"
    ]
    assert "don't rank attorneys" in result["generation"]
    assert result["sources"]
    assert result["grounding_verdict"] == "pass"


def test_pure_refusal_hides_unrelated_sources() -> None:
    settings = make_settings()
    doc = Document(
        page_content="Employment counseling and office locations are described here.",
        metadata={"source_file": "policy.md", "section_heading": "Employment", "page_number": 1},
    )
    llm = FakeLLM(
        [
            "I couldn't find a remote-work policy in the indexed documents.",
        ]
    )
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever, query="What is the remote work policy?")

    assert result["sources"] == []
    assert result["citations"] == []
    assert result["citation_warnings"] == []
    assert result["grounding_verdict"] == "pass"


def test_grounding_failure_retries_generation() -> None:
    settings = make_settings(max_grounding_retries=1)
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    llm = FakeLLM(["rag", "Unsupported answer", "no", "Grounded answer", "yes"])
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever)

    assert result["generation"] == "Grounded answer"
    assert result["grounding_retry_count"] == 1
    assert result["grounding_verdict"] == "pass"


def test_multi_turn_condensation_uses_checkpointed_history() -> None:
    settings = make_settings()
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    # The first question is routed by the deterministic firm-term heuristic,
    # so it does not consume an LLM response. The second needs an LLM route.
    llm = FakeLLM(
        [
            "First answer",
            "yes",
            "rag",
            "standalone follow-up",
            "Second answer",
            "yes",
        ]
    )
    retriever = FakeRetriever([[doc], [doc]])

    deps = GraphDependencies(
        settings=settings,
        llm=llm,
        retriever=retriever,
        reranker=FakeReranker(),
    )
    graph = build_graph(deps)
    config = {"configurable": {"thread_id": "multi-turn"}, "recursion_limit": 30}

    first = graph.invoke(
        {"messages": [HumanMessage("What is the consultation rate?")]}, config=config
    )
    second = graph.invoke({"messages": [HumanMessage("What about filings?")]}, config=config)

    assert first["generation"] == "First answer"
    assert second["generation"] == "Second answer"
    assert retriever.queries == ["What is the consultation rate?", "standalone follow-up"]


def test_default_grader_mode_skips_llm_relevance_calls() -> None:
    settings = make_settings()
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing"},
    )
    # route -> generate -> grounding. No per-document "yes" grade response.
    llm = FakeLLM(["rag", "Consultations cost $300.", "yes"])
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever)

    assert result["generation"] == "Consultations cost $300."
    assert result["grounding_verdict"] == "pass"


def test_sqlite_checkpointer_creates_database(tmp_path) -> None:
    from graph import create_checkpointer

    settings = make_settings(
        checkpoint_backend="sqlite",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite",
    )
    checkpointer = create_checkpointer(settings)

    assert (tmp_path / "checkpoints.sqlite").exists()
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        connection.close()


def test_out_of_scope_routes_away_from_retrieval() -> None:
    settings = make_settings()
    llm = FakeLLM(["out_of_scope", "I can only help with firm knowledge."])
    retriever = FakeRetriever([])

    result = invoke_graph(settings, llm, retriever)

    assert result["route_decision"] == "out_of_scope"
    assert retriever.queries == []
    assert result["generation"] == "I can only help with firm knowledge."
