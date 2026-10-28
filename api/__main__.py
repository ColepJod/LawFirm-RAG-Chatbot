"""Run the API locally: ``python -m api``."""

from __future__ import annotations

import uvicorn

from config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
