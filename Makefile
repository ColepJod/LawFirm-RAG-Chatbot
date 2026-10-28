# Local development tasks. `uv` is required.
# On Windows without GNU Make, run the equivalent `uv run ...` command directly.

.PHONY: install lock config config-dirs test lint check

install:
	uv sync --extra dev

lock:
	uv lock

config:
	uv run python -m config

config-dirs:
	uv run python -m config --check-dirs --show-cache-env

ingest:
	uv run python -m ingestion --path data

serve:
	uv run python -m api

test:
	uv run pytest -q

lint:
	uv run ruff check .

check: lint test
