"""Checkpoint factories for LangGraph conversation memory."""

from __future__ import annotations

from config import get_settings


def create_checkpointer(settings=None, persistent: bool | None = None):
    """Create an in-memory or SQLite-backed LangGraph checkpointer."""

    settings = settings or get_settings()
    use_persistent = settings.checkpoint_backend == "sqlite" if persistent is None else persistent

    if use_persistent:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("SQLite checkpointing requires langgraph-checkpoint-sqlite") from exc

        settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(settings.checkpoint_db_path), check_same_thread=False)
        return SqliteSaver(connection)

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
