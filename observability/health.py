"""Health checks for the local application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import get_settings


def health_check(
    settings=None,
    vector_store=None,
    bm25_index=None,
    ingestion_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly health report."""

    settings = settings or get_settings()
    checks: dict[str, Any] = {}
    healthy = True

    checks["config"] = {"status": "ok", "environment": settings.environment}

    # Vector store connectivity.
    try:
        if vector_store is None:
            checks["vector_store"] = {"status": "not_configured"}
        else:
            vector_store.similarity_search("health check", k=1)
            checks["vector_store"] = {"status": "ok", "backend": settings.vector_store_backend}
    except Exception as exc:
        healthy = False
        checks["vector_store"] = {"status": "error", "error": str(exc)}

    # BM25 index.
    if bm25_index is None:
        checks["bm25"] = {"status": "not_configured"}
    else:
        checks["bm25"] = {"status": "ok", "document_count": bm25_index.document_count()}

    # Last ingestion timestamp.
    manifest_path = ingestion_manifest_path or (
        settings.project_root / "var" / "ingestion" / "manifest.json"
    )
    last_ingestion = None
    if manifest_path.exists():
        import json

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            timestamps = [
                value.get("ingestion_timestamp")
                for value in data.values()
                if isinstance(value, dict)
            ]
            last_ingestion = max((ts for ts in timestamps if ts), default=None)
        except Exception:
            last_ingestion = None
    checks["ingestion"] = {"status": "ok", "last_ingestion_timestamp": last_ingestion}

    # Model cache.
    cache_ok = settings.local_model_cache_dir.exists() and any(
        settings.local_model_cache_dir.iterdir()
    )
    checks["model_cache"] = {
        "status": "ok" if cache_ok else "empty",
        "path": str(settings.local_model_cache_dir),
    }

    return {"status": "ok" if healthy else "degraded", "checks": checks}
