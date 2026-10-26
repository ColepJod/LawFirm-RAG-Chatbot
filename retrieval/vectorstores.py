"""Vector-store backends behind a small, swappable abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from config import get_settings


class VectorStoreBackend(Protocol):
    """Minimal interface required by ingestion and retrieval code."""

    backend_name: str
    collection_name: str

    def add_documents(self, documents, ids=None): ...

    def delete(self, ids): ...

    def as_retriever(self, k: int | None = None): ...

    def similarity_search(self, query: str, k: int = 4): ...


@dataclass
class VectorStoreHandle:
    """A LangChain vector store plus backend metadata."""

    backend_name: str
    collection_name: str
    store: Any

    def add_documents(self, documents, ids=None):
        return self.store.add_documents(documents, ids=ids)

    def delete(self, ids):
        return self.store.delete(ids)

    def as_retriever(self, k: int | None = None):
        search_kwargs = {"k": k} if k is not None else {}
        return self.store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 4):
        return self.store.similarity_search(query, k=k)


def create_vector_store(settings=None, embeddings=None):
    """Create the configured vector-store backend."""

    settings = settings or get_settings()
    if embeddings is None:
        from retrieval.embeddings import create_embeddings

        embeddings = create_embeddings(settings)

    collection_name = settings.vector_store_collection

    if settings.vector_store_backend == "chroma":
        from langchain_chroma import Chroma

        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )
        return VectorStoreHandle("chroma", collection_name, store)

    if settings.vector_store_backend == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_local_path))

        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return VectorStoreHandle("qdrant", collection_name, store)

    raise ValueError(f"Unsupported vector store backend: {settings.vector_store_backend}")
