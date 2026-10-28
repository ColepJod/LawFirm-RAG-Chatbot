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
    def compress_documents(self, documents, query):
        return list(documents)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def invoke_graph(
    settings: Settings, llm: FakeLLM, retriever: FakeRetriever, **deps_overrides
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
    return graph.invoke({"messages": [HumanMessage("test query")]}, config=config)


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
    llm = FakeLLM(["rag", "yes", "Consultations cost $300.", "yes"])
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever)

    assert result["route_decision"] == "rag"
    assert retriever.queries == ["test query"]
    assert result["generation"] == "Consultations cost $300."
    assert result["grounding_verdict"] == "pass"
    assert result["citations"]


def test_correction_rewrites_and_retrieves_again() -> None:
    settings = make_settings()
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

    result = invoke_graph(settings, llm, retriever)

    assert retriever.queries == ["test query", "better billing question"]
    assert result["generation"] == "Consultations cost $300."
    assert result["rewrite_count"] == 1


def test_empty_retrieval_refuses_without_rewrite_when_disabled() -> None:
    settings = make_settings(max_query_rewrites=0)
    llm = FakeLLM(["rag"])
    retriever = FakeRetriever([[]])

    result = invoke_graph(settings, llm, retriever)

    assert result["retrieved_chunks"] == []
    assert "do not have enough information" in result["generation"].lower()
    assert result["grounding_verdict"] == "pass"


def test_grounding_failure_retries_generation() -> None:
    settings = make_settings(max_grounding_retries=1)
    doc = Document(
        page_content="Consultations cost $300.",
        metadata={"source_file": "rates.md", "section_heading": "Billing", "page_number": 1},
    )
    llm = FakeLLM(["rag", "yes", "Unsupported answer", "no", "Grounded answer", "yes"])
    retriever = FakeRetriever([[doc]])

    result = invoke_graph(settings, llm, retriever)

    assert result["generation"] == "Grounded answer"
    assert result["grounding_retry_count"] == 1
    assert result["grounding_verdict"] == "pass"


def test_out_of_scope_routes_away_from_retrieval() -> None:
    settings = make_settings()
    llm = FakeLLM(["out_of_scope", "I can only help with firm knowledge."])
    retriever = FakeRetriever([])

    result = invoke_graph(settings, llm, retriever)

    assert result["route_decision"] == "out_of_scope"
    assert retriever.queries == []
    assert result["generation"] == "I can only help with firm knowledge."
