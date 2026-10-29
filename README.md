# Law Firm RAG Chatbot

A local-first, multi-document Retrieval-Augmented Generation (RAG) chatbot for a
law firm. It is designed for an arbitrary corpus of PDF, Markdown, plain-text,
and DOCX documents, not a single PDF.

The target stack is free, open-source, and locally runnable:

- **Orchestration:** LangGraph `StateGraph` + LangChain
- **Parsing:** pdfmux, Unstructured OSS, Docling
- **Vector store:** ChromaDB by default, Qdrant optional
- **Embeddings:** local Hugging Face models (no API key)
- **Sparse retrieval:** BM25 via `rank_bm25`
- **Reranking:** local cross-encoder
- **LLM:** Groq by default, with Ollama and vLLM still supported
- **Evaluation:** llamella + ragnarok-ai, with optional local RAGAS
- **Observability:** structured JSON traces, optional self-hosted Langfuse

No cloud API key is required.

## Current status

**Steps 1-12 are complete.** The repository contains the full application:
project skeleton, typed configuration, pinned and locked dependencies,
multi-format ingestion, configurable chunking, local embedding/indexing with
Chroma or Qdrant plus BM25, incremental indexing, Reciprocal Rank Fusion, local
cross-encoder reranking, a LangGraph RAG control loop, checkpointed multi-turn
memory, a local-first evaluation harness, observability and guardrails, a
FastAPI layer, and unit plus end-to-end tests.

No Docker, Kubernetes, Terraform, CI/CD, or cloud deployment files are included.

## Web UI

After starting the API:

```powershell
uv run python -m api
```

Open:

```text
http://127.0.0.1:8000/      Web chat UI
http://127.0.0.1:8000/docs   Interactive API docs
```

The API and end-to-end tests are implemented in later steps.

## Requirements

- Python 3.13 (3.11 and 3.12 are also supported by the declared constraints)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Ollama for local generation (installed separately; no model is downloaded by
  this repository)
- Internet access on first model use, to download open-weight models from
  Hugging Face and the Ollama registry

On this machine, Ollama is installed, but only `:cloud` models are currently
present. A local open-weight model must be pulled before chat can run in a later
step. No model was downloaded during Step 1.

## Install (Step 1 scope)

```powershell
uv venv --python 3.13
uv pip install -e ".[dev]"
```

Or, after a lockfile exists:

```powershell
uv sync --extra dev
```

Step 1 installs only the base configuration dependencies plus development tools.
The heavier extras are declared but are installed in the steps that need them:

```text
orchestration  LangChain, LangGraph, LangChain community/text splitters
ingest         pdfmux, Docling, Unstructured
retrieval      Chroma, Qdrant client, sentence-transformers, BM25, tiktoken
generation     langchain-ollama, Ollama Python client
evaluation     llamella, ragnarok-ai, RAGAS
api            FastAPI, Uvicorn, python-multipart
observability  Langfuse OSS client
dev            pytest, Ruff, coverage, HTTPX
```

## Configure

```powershell
Copy-Item .env.example .env
```

Then inspect the validated configuration:

```powershell
uv run python -m config
uv run python -m config --check-dirs --show-cache-env
```

Relative paths are resolved against the repository root. `LOCAL_MODEL_CACHE_DIR`
controls where Hugging Face and sentence-transformers place model weights.

### Hardware profiles

`RESOURCE_PROFILE` supplies defaults only when a model variable is not set:

| Profile | Embedding | Reranker | LLM |
|---|---|---|---|
| `standard` | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | `qwen3:8b` |
| `cpu` | `BAAI/bge-small-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `qwen3:4b` |

Explicit values such as `EMBEDDING_MODEL` always override the profile. The
profile never switches a model silently after an index has been built, because
changing the embedding model invalidates the vector index.

The example `.env` selects the `cpu` profile because this development machine
has no CUDA-capable GPU.

## Why `sentence-transformers==3.4.1`?

`llamella 0.1.0` — the required deterministic response-quality scorer — declares
`sentence-transformers>=3.0,<4.0`. The current upstream release is `6.2.0`.
Step 1 pins `3.4.1` so the runtime and evaluation stack share one environment.
Both required embedding and reranker families still work with that version.

This is a deliberate compatibility pin, not a claim that `3.4.1` is the newest
release. The evaluation layer will be kept behind an adapter so the dependency
can be upgraded or isolated later.

## Project structure

```text
config/        Typed settings, model profiles, cache bootstrap, config CLI
ingestion/     PDF/Markdown/TXT/DOCX parsing and chunking (Steps 2-3)
retrieval/     Embeddings, vector store, BM25, RRF, reranker (Steps 4-6)
graph/         LangGraph state and nodes (Steps 7-8)
generation/    Grounded answer generation (Step 7)
evaluation/    Golden set and local evaluation harness (Step 9)
api/           FastAPI application (Step 11)
tests/         Unit and integration tests
data/          Source documents (gitignored)
var/           Chroma, logs, traces, uploads, evaluation output (gitignored)
```

## Configuration principles

1. **Fail fast.** Invalid overlap, unsupported backends, and impossible retrieval
   budgets raise a validation error at startup.
2. **One source of truth.** The CLI, API, ingestion, and evaluation all read the
   same `Settings` object.
3. **No hidden cloud dependency.** No variable requires a paid API key. Optional
   keys exist only for a self-hosted Langfuse instance or an optional vLLM
   deployment.
4. **Secrets stay out of logs.** Optional keys use Pydantic `SecretStr` and are
   redacted by the config CLI.
5. **Paths are project-relative.** Starting the API from another directory does
   not create a second empty Chroma database.

## Common commands

```powershell
uv run python -m config       # show validated config
uv run pytest -q              # run tests
uv run ruff check .           # lint
uv lock                       # resolve the full dependency set
```

The `Makefile` provides the same commands where GNU Make is available:
`make config`, `make test`, `make lint`, `make check`.

The required `make ingest`, `make chat`, and `make eval` targets will be added
when the corresponding steps are implemented. They are intentionally not stubbed
yet, because a target that silently does nothing would be misleading.

## Roadmap

| Step | Scope |
|---|---|
| 1 | Skeleton and typed configuration (complete) |
| 2 | Multi-format ingestion |
| 3 | Configurable chunking strategies |
| 4 | Embeddings, indexing, dense + BM25 retrievers |
| 5 | Hybrid retrieval with RRF |
| 6 | Local cross-encoder reranking |
| 7 | LangGraph RAG control loop |
| 8 | Checkpointed conversation memory |
| 9 | Local-first evaluation harness |
| 10 | JSON traces and guardrails |
| 11 | FastAPI layer |
| 12 | Tests across the pipeline |

No Docker, Kubernetes, Terraform, CI/CD, or cloud deployment files are included.

## Web UI

After starting the API:

```powershell
uv run python -m api
```

Open:

```text
http://127.0.0.1:8000/      Web chat UI
http://127.0.0.1:8000/docs   Interactive API docs
```
