"""Prompt templates used by the LangGraph nodes."""

from __future__ import annotations

ROUTE_PROMPT = """Classify the user's message for a law-firm knowledge assistant.

Return exactly one of these labels and nothing else:
- rag: the user asks about documents, rates, locations, specialties, billing, or comparisons
- chit_chat: greetings, thanks, or conversational small talk
- out_of_scope: unrelated to the firm or its documents

User message: {query}
Label:"""

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question.
Use the conversation history only to resolve pronouns, ellipsis, or references.
Do not answer the question.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:"""

REWRITE_PROMPT = """The previous retrieval did not find relevant documents.
Rewrite this question to improve retrieval. Keep it concise and specific.
Return only the rewritten question.

Question: {query}
Rewritten question:"""

GRADE_PROMPT = """You are a relevance judge for a law-firm knowledge assistant.

Return a JSON object with a single boolean field named "relevant".
Use true when the document helps answer the question, false otherwise.

Question: {query}

Document:
{document}"""

GENERATE_PROMPT = """You are a law-firm RAG assistant. Answer only from the provided context.
If the context is insufficient, say that you do not have enough information.
Cite every factual statement using the document, section, and page information in the context.

Question: {query}

Context:
{context}

Answer:"""

GROUNDING_PROMPT = """You are a grounding judge for a law-firm knowledge assistant.

Return a JSON object with a single boolean field named "grounded".
Use true when the answer is fully supported by the context, false otherwise.

Context:
{context}

Answer:
{answer}"""

DIRECT_ANSWER_PROMPT = """You are a helpful assistant for a law firm. Answer briefly.
If the user asks about firm documents, rates, locations, specialties, or billing,
say that they should ask a knowledge question instead.

User message: {query}
Answer:"""
