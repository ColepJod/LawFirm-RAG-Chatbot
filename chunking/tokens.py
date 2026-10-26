"""Token counting used by the chunking layer."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def get_encoding():
    """Return the tokenizer used to enforce chunk budgets."""

    # cl100k_base is a stable, widely available approximation. Embedding
    # tokenizers differ slightly, so chunk budgets remain conservative.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in ``text``."""

    if not text:
        return 0
    return len(get_encoding().encode(text))


def token_window_split(text: str, max_tokens: int) -> list[str]:
    """Split text into fixed token windows as a last-resort fallback."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    encoding = get_encoding()
    tokens = encoding.encode(text)
    if not tokens:
        return []

    windows: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        windows.append(encoding.decode(tokens[start : start + max_tokens]))
    return windows
