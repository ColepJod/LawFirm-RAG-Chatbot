"""Deterministic response-quality scoring for offline evaluation.

The original harness delegated these metrics to llamella, which downloads a
DeBERTa NLI cross-encoder (hundreds of megabytes) the first time it scores an
answer. That makes evaluation slow, network-dependent, and awkward in CI.

These metrics are lexical proxies. They are intentionally simple and fully
deterministic: token coverage, expected-answer recall, query coverage, and
citation presence. They are useful for regression detection and for catching
answers that drift away from retrieved context, but they are not a substitute
for human review of legal-answer quality.
"""

from __future__ import annotations

import re
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
CITATION_PATTERN = re.compile(r"\[\s*\d+[^\]]*\]")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Common English function words. Legal terms, numbers, and names are retained.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall((text or "").lower())
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    }


def _coverage(source: set[str], reference: set[str]) -> float:
    if not source:
        return 0.0
    return len(source & reference) / len(source)


def _recall(expected: set[str], answer: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & answer) / len(expected)


def _sentence_support(sentences: list[str], context_tokens: set[str]) -> float:
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        sentence_tokens = _tokens(sentence)
        if not sentence_tokens:
            continue
        # Generated prose adds connectives and paraphrases, so require 40% of a
        # sentence's content tokens to appear in the retrieved context.
        if _coverage(sentence_tokens, context_tokens) >= 0.4:
            supported += 1
    return supported / len(sentences)


def _citation_ratio(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    cited = sum(1 for sentence in sentences if CITATION_PATTERN.search(sentence))
    return cited / len(sentences)


def score_response(
    query: str,
    response: str,
    contexts: list[str],
    *,
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """Score one response against its query, contexts, and expected answer.

    Returns groundedness, completeness, relevance, consistency, confidence, and
    an overall index (IQS) in the 0-1 range. Extra diagnostic fields are
    included for inspection but are not part of the threshold checks.
    """

    answer = response or ""
    sentences = [
        sentence.strip() for sentence in SENTENCE_PATTERN.split(answer) if sentence.strip()
    ]
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts or []))
    query_tokens = _tokens(query)
    expected_tokens = _tokens(expected_answer or "")

    if not answer_tokens:
        return {
            "groundedness": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "consistency": 0.0,
            "confidence": 0.0,
            "iqs": 0.0,
            "answer_tokens": 0,
            "supported_sentences": 0.0,
            "citation_ratio": 0.0,
        }

    token_grounding = _coverage(answer_tokens, context_tokens)
    sentence_grounding = _sentence_support(sentences, context_tokens)
    groundedness = round(0.5 * token_grounding + 0.5 * sentence_grounding, 4)

    completeness = (
        round(_recall(expected_tokens, answer_tokens), 4) if expected_tokens else groundedness
    )

    if query_tokens:
        answer_relevance = _coverage(query_tokens, answer_tokens)
        context_relevance = _coverage(query_tokens, context_tokens)
        relevance = round(0.7 * answer_relevance + 0.3 * context_relevance, 4)
    else:
        relevance = 1.0

    citations = _citation_ratio(sentences)
    # Lexical metrics cannot detect logical contradictions. Use grounding as
    # the consistency proxy and apply a small penalty when no citation is shown.
    consistency = round(groundedness * (1.0 if citations else 0.9), 4)

    confidence = round(0.5 * groundedness + 0.3 * completeness + 0.2 * relevance, 4)
    iqs = round((groundedness + completeness + relevance + consistency) / 4.0, 4)

    return {
        "groundedness": groundedness,
        "completeness": completeness,
        "relevance": relevance,
        "consistency": consistency,
        "confidence": confidence,
        "iqs": iqs,
        "answer_tokens": len(answer_tokens),
        "supported_sentences": round(sentence_grounding, 4),
        "citation_ratio": round(citations, 4),
    }
