"""Small, dependency-free security helpers for the API layer."""

from __future__ import annotations

import re
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

from config import Settings
from ingestion.discovery import SUPPORTED_EXTENSIONS

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def sanitize_filename(filename: str | None) -> str:
    """Return a safe basename that cannot escape the upload directory."""

    raw = Path(filename or "uploaded_document").name
    raw = raw.replace("\\", "_").replace("/", "_")
    raw = SAFE_FILENAME_RE.sub("_", raw).strip(" ._")
    if not raw:
        raw = "uploaded_document"
    suffix = Path(raw).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_file_type",
                "message": f"Unsupported file type: {suffix or 'none'}",
                "allowed": sorted(SUPPORTED_EXTENSIONS),
            },
        )
    stem = Path(raw).stem[:120] or "document"
    return f"{stem}{suffix}"


def normalize_conversation_id(value: str | None) -> str:
    """Validate an explicit conversation ID or create a fresh UUID."""

    if value is None or not value.strip():
        return str(uuid.uuid4())
    candidate = value.strip()
    if not CONVERSATION_ID_RE.match(candidate):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_conversation_id",
                "message": "conversation_id must be 1-128 characters of letters, digits, . _ : -",
            },
        )
    return candidate


def _provided_admin_token(request: Request) -> str | None:
    token = request.headers.get("x-admin-token")
    if token:
        return token
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def admin_allowed(request: Request, settings: Settings) -> bool:
    expected = settings.admin_api_token
    if expected is None:
        # Local development remains frictionless. Staging/production must set
        # ADMIN_API_TOKEN before exposing ingestion or document management.
        return settings.environment in {"local", "test"}

    provided = _provided_admin_token(request)
    if not provided:
        return False
    return secrets.compare_digest(provided, expected.get_secret_value())


def require_admin(request: Request, settings: Settings) -> None:
    if not admin_allowed(request, settings):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "admin_auth_required",
                "message": "A valid admin token is required for this endpoint.",
            },
        )


class TokenBucketRateLimiter:
    """Thread-safe in-memory token bucket keyed by client or conversation."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(1, per_minute)
        self.capacity = float(self.per_minute)
        self.refill_per_second = self.per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, updated = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.refill_per_second)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            self._maybe_cleanup(now)
            return allowed

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        stale = [
            key
            for key, (tokens, updated) in self._buckets.items()
            if tokens >= self.capacity and now - updated > 300
        ]
        for key in stale:
            self._buckets.pop(key, None)


def client_key(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host or "unknown"


async def save_upload_limited(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Stream an upload to disk while enforcing a hard size limit."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    total = 0
    try:
        with temporary.open("wb") as stream:
            while True:
                block = await upload.read(chunk_size)
                if not block:
                    break
                total += len(block)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "upload_too_large",
                            "message": f"Upload exceeds the {max_bytes} byte limit.",
                        },
                    )
                stream.write(block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return total
