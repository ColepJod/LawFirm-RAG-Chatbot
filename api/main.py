"""FastAPI application for the law-firm RAG chatbot."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.factory import build_app_state
from api.routes import router
from api.state import AppState

app_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_state
    app_state = build_app_state()
    yield
    app_state = None


app = FastAPI(
    title="Law Firm RAG Chatbot",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


def get_app_state() -> AppState:
    if app_state is None:
        raise RuntimeError("Application state is not initialized")
    return app_state
