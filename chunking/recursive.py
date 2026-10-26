"""Recursive token-based chunking with table-row protection."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.context import build_context_header, resolve_section_heading
from chunking.tables import split_into_blocks, split_table_block
from chunking.tokens import count_tokens, token_window_split


class RecursiveTokenChunker:
    """Chunk text recursively while never splitting a markdown table row."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 96,
        min_chunk_chars: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.strategy = "recursive"

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for document in documents:
            result.extend(self._split_document(document))
        return result

    def raw_chunks(self, text: str, budget_tokens: int) -> list[str]:
        """Return text chunks without context headers."""

        chunks: list[str] = []
        for block_type, block_text in split_into_blocks(text):
            if not block_text.strip():
                continue
            if block_type == "table":
                chunks.extend(split_table_block(block_text, budget_tokens))
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=budget_tokens,
                    chunk_overlap=self.chunk_overlap,
                    length_function=count_tokens,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks.extend(splitter.split_text(block_text))

        filtered = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= self.min_chunk_chars]
        if not filtered and text.strip():
            filtered = [text.strip()]

        # Last-resort guarantee: no chunk may exceed the configured token budget.
        guaranteed: list[str] = []
        for chunk in filtered:
            if count_tokens(chunk) <= self.chunk_size:
                guaranteed.append(chunk)
            else:
                guaranteed.extend(token_window_split(chunk, budget_tokens))
        return guaranteed

    def _split_document(self, document: Document) -> list[Document]:
        section_heading = resolve_section_heading(document)
        header = build_context_header(document, section_heading)
        header_tokens = count_tokens(header)
        budget = max(1, self.chunk_size - header_tokens - 2)

        raw = self.raw_chunks(document.page_content, budget)
        if not raw:
            raw = [document.page_content.strip()] if document.page_content.strip() else []

        chunks: list[Document] = []
        for index, chunk_text in enumerate(raw):
            content = f"{header}\n{chunk_text.strip()}"
            metadata = dict(document.metadata or {})
            metadata["section_heading"] = section_heading
            metadata["chunk_strategy"] = self.strategy
            metadata["chunk_index"] = index
            metadata["chunk_token_count"] = count_tokens(content)
            metadata["chunk_id"] = (
                f"{metadata.get('document_hash', 'document')}:{self.strategy}:{index}"
            )
            chunks.append(Document(page_content=content, metadata=metadata))

        if not chunks:
            content = f"{header}\n"
            metadata = dict(document.metadata or {})
            metadata["section_heading"] = section_heading
            metadata["chunk_strategy"] = self.strategy
            metadata["chunk_index"] = 0
            metadata["chunk_token_count"] = count_tokens(content)
            metadata["chunk_id"] = f"{metadata.get('document_hash', 'document')}:{self.strategy}:0"
            chunks.append(Document(page_content=content, metadata=metadata))

        return chunks
