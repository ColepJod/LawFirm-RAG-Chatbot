from __future__ import annotations

from collections.abc import Iterator

import pytest

from config import Settings, get_settings

CACHE_ENV_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_DISABLE_TELEMETRY",
    "SENTENCE_TRANSFORMERS_HOME",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep unit tests independent of the developer's shell environment."""

    for field_name in Settings.model_fields:
        monkeypatch.delenv(field_name.upper(), raising=False)
    for name in CACHE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
