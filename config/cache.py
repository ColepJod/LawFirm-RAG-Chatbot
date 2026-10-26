"""Hugging Face and sentence-transformers cache bootstrapping.

These environment variables must be set before model libraries are imported.
If they are set later, libraries may download the same multi-gigabyte model a
second time into a different cache.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


def apply_model_cache_environment(settings: Settings | None = None) -> dict[str, str]:
    """Point local model caches at ``LOCAL_MODEL_CACHE_DIR``.

    Existing process environment values are respected. Returns the effective
    values so they can be logged by the configuration check command.
    """

    if settings is None:
        from config.settings import get_settings

        settings = get_settings()

    cache_root = settings.local_model_cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)

    values = {
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "SENTENCE_TRANSFORMERS_HOME": str(cache_root / "sentence-transformers"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "TOKENIZERS_PARALLELISM": "false",
    }

    for key, value in values.items():
        os.environ.setdefault(key, value)

    return {key: os.environ[key] for key in values}
