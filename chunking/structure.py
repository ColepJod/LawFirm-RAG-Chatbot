"""Heading-aware chunking with recursive fallback for large sections."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from chunking.context import build_context_header, resolve_section_heading
from chunking.recursive import RecursiveTokenChunker
from chunking.tokens import count_tokens

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


class StructureAwareChunker:
    """Split Markdown-like documents by headings, then by token budget."""

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
        self.strategy = "structure"
        self._recursive = RecursiveTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for document in documents:
            result.extend(self._split_document(document))
        return result

    def _split_document(self, document: Document) -> list[Document]:
        text = document.page_content or ""
        if not re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
            recursive_chunks = self._recursive._split_document(document)
            for chunk in recursive_chunks:
                chunk.metadata["chunk_strategy"] = self.strategy
                chunk.metadata["chunk_id"] = str(chunk.metadata["chunk_id"]).replace(
                    ":recursive:", ":structure:"
                )
            return recursive_chunks

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=True,
        )
        sections = splitter.split_text(text)
        if not sections:
            sections = [Document(page_content=text, metadata={})]

        chunks: list[Document] = []
        for section in sections:
            section_heading = self._section_heading(section.metadata, document)
            header = build_context_header(document, section_heading)
            header_tokens = count_tokens(header)
            budget = max(1, self.chunk_size - header_tokens - 2)
            section_text = (section.page_content or "").strip()

            if not section_text:
                continue

            if count_tokens(header + "\n" + section_text) <= self.chunk_size:
                raw_chunks = [section_text]
            else:
                raw_chunks = self._recursive.raw_chunks(section_text, budget)
                if not raw_chunks:
                    raw_chunks = [section_text]

            for raw_chunk in raw_chunks:
                content = f"{header}\n{raw_chunk.strip()}"
                metadata = dict(document.metadata or {})
                metadata.update(section.metadata or {})
                metadata["section_heading"] = section_heading
                metadata["chunk_strategy"] = self.strategy
                metadata["chunk_index"] = len(chunks)
                metadata["chunk_token_count"] = count_tokens(content)
                metadata["chunk_id"] = (
                    f"{metadata.get('document_hash', 'document')}:{self.strategy}:{len(chunks)}"
                )
                chunks.append(Document(page_content=content, metadata=metadata))

        if not chunks:
            fallback = self._recursive._split_document(document)
            for chunk in fallback:
                chunk.metadata["chunk_strategy"] = self.strategy
            return fallback

        return chunks

    @staticmethod
    def _section_heading(section_metadata: dict[str, object], document: Document) -> str:
        for key in ("h4", "h3", "h2", "h1"):
            value = section_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return resolve_section_heading(document)
