"""FastAPI application for Law Help."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.factory import build_app_state
from api.routes import router
from api.state import AppState
from config import get_settings

logger = logging.getLogger(__name__)

app_state: AppState | None = None

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    app_state = build_app_state()
    logger.info(
        "application ready index_fingerprint=%s chunks=%d",
        app_state.index_fingerprint,
        app_state.bm25_index.document_count(),
    )
    try:
        yield
    finally:
        _shutdown_state(app_state)
        app_state = None


def _shutdown_state(state: AppState | None) -> None:
    if state is None:
        return
    checkpointer = state.checkpointer
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        try:
            connection.close()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.debug("failed to close checkpoint connection", exc_info=True)


app = FastAPI(
    title="Law Help",
    description="Citation-backed answers from a law firm's indexed documents.",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
_origins = [origin.strip() for origin in _settings.cors_origins.split(",") if origin.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Token", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
    )


@app.middleware("http")
async def security_and_logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if request.url.path.startswith("/static/"):
        # Assets are not fingerprinted, so make browsers revalidate them.
        response.headers.setdefault("Cache-Control", "no-cache")
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
UI_PATH = STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(UI_PATH)


@app.get("/ui", include_in_schema=False)
async def ui_alias() -> FileResponse:
    return FileResponse(UI_PATH)


def get_app_state() -> AppState:
    if app_state is None:
        raise RuntimeError("Application state is not initialized")
    return app_state
