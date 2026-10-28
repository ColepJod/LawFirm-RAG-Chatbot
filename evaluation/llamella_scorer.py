"""Deterministic response-quality scoring with llamella."""

from __future__ import annotations

import contextlib
from typing import Any

try:
    import llamella
except ImportError as exc:  # pragma: no cover - dependency is declared
    raise RuntimeError("llamella is required for evaluation") from exc


def score_response(
    query: str,
    response: str,
    contexts: list[str],
    *,
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    embedding_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> dict[str, Any]:
    """Score one response with llamella and return a plain dictionary."""

    with contextlib.suppress(Exception):
        llamella.setup_nltk()

    result = llamella.score(
        query=query,
        response=response,
        context=contexts,
        nli_model=nli_model,
        embedding_model=embedding_model,
        device=device,
    )
    return result.to_dict()
