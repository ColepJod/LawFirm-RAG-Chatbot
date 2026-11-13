# Law Help

Law Help is a production-oriented, local-first Retrieval-Augmented Generation (RAG)
assistant for a law firm knowledge base. It ingests PDF, Markdown, plain-text, and
DOCX documents, retrieves evidence with hybrid search, reranks it locally, and
returns answers with citation-labelled sources.

The project is designed for a **single hardened node**: one FastAPI process, one
SQLite conversation store, one local Chroma collection, and one local BM25 index.
That constraint keeps the system easy to audit and operate without Redis,
Postgres, Elasticsearch, or a container platform. The boundaries that matter for
scaling are documented in [Operations](#operations).

- **Orchestration:** LangGraph `StateGraph` + LangChain
- **Parsing:** pdfmux, Unstructured, Docling
- **Vector store:** Chroma by default; Qdrant optional
- **Embeddings:** local Hugging Face models (`bge-small-en-v1.5` on CPU)
- **Sparse retrieval:** in-process BM25
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranking:** local cross-encoder (`ms-marco-MiniLM-L-6-v2` on CPU)
- **LLM:** Groq, Ollama, or any OpenAI-compatible vLLM endpoint
- **API:** FastAPI with SSE streaming, jobs, and document management
- **UI:** zero-build HTML/CSS/ES modules served by FastAPI
- **Evaluation:** deterministic lexical metrics over a corpus-derived golden set
- **Observability:** JSON logs, JSONL traces, health and metrics endpoints

No frontend build step and no cloud API key are required. A hosted Groq key is
optional; Ollama and vLLM run entirely on your hardware.

---

## Table of contents

1. [What the system does](#what-the-system-does)
2. [Architecture](#architecture)
3. [Request lifecycle](#request-lifecycle)
4. [Project layout](#project-layout)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [How the first step was built: skeleton and typed config](#how-the-first-step-was-built-skeleton-and-typed-config)
8. [Frontend and backend connection](#frontend-and-backend-connection)
9. [API reference](#api-reference)
10. [Ingestion and indexing](#ingestion-and-indexing)
11. [Retrieval pipeline](#retrieval-pipeline)
12. [Graph control loop](#graph-control-loop)
13. [Library choices and alternatives](#library-choices-and-alternatives)
14. [Security and guardrails](#security-and-guardrails)
15. [Evaluation](#evaluation)
16. [Operations](#operations)
17. [Testing and quality gates](#testing-and-quality-gates)
18. [Troubleshooting](#troubleshooting)
19. [Current limitations](#current-limitations)

---

## What the system does

The assistant answers questions such as:

- “What are the firm's standard partner rates?”
- “Who leads the Intellectual Property practice?”
- “Where is the Chicago office and who manages it?”
- “How long are closed matter files retained?”
- “What does a blended rate mean?”

Every answer is grounded in retrieved chunks. The response includes:

- the generated answer with bracketed citation labels (`[1]`, `[1, 2]`);
- a structured `sources` array with file, section, page, score, and snippet;
- warnings when citation coverage is imperfect;
- the grounding verdict and per-node latencies;
- a stable `thread_id` for conversation continuity.

The system deliberately refuses unsupported answers. If retrieval returns no
documents for a knowledge question, the graph either rewrites the query or
returns a grounded refusal rather than inventing an answer.

---

## Architecture

### Runtime topology

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Browser                                                              │
│  index.html + styles.css + app.js (no build step)                    │
│  fetch('/chat/stream') ──► SSE progress + final answer               │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ same origin
┌───────────────────────────────────▼──────────────────────────────────┐
│ FastAPI (single worker)                                              │
│  api/main.py       security headers, CORS, static UI, lifespan       │
│  api/routes.py     chat, SSE, ingest, jobs, documents, health        │
│  api/security.py   admin token, upload limits, sanitization, limits  │
│  api/factory.py    model/store/graph construction and warmup         │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│ LangGraph control loop                                               │
│  route → condense → retrieve → grade → rerank/rewrite → generate     │
│        → check_grounding → log_trace                                 │
└───────────────┬──────────────────────────────┬───────────────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼──────────────────────┐
│ Hybrid retrieval             │  │ Generation                        │
│  Chroma dense search         │  │  Groq / Ollama / vLLM             │
│  BM25 sparse search          │  │  citation prompt + grounding      │
│  RRF fusion                  │  │  retry                            │
│  cross-encoder rerank        │  └───────────────────────────────────┘
│  LRU retrieval/rerank caches │
└───────────────┬──────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ Persistence                                                          │
│  var/chroma/          vector collection                              │
│  var/index/           chunks.jsonl + schema-v2 manifest             │
│  var/ingestion/       ingestion manifest (hash idempotency)          │
│  var/checkpoints.sqlite  LangGraph conversation checkpoints          │
│  var/traces/          JSONL run traces                               │
│  var/uploads/         per-job uploaded files                          │
└──────────────────────────────────────────────────────────────────────┘
```

### Data flow at ingest time

```text
data/ or upload
   │
   ▼
discover supported files (.pdf .md .markdown .txt .docx)
   │
   ▼
parse into LangChain Documents (page/metadata/section aware)
   │
   ▼
structure-aware chunking with token budgets
   │
   ├──► local embedding model ──► Chroma (dense vectors)
   ├──► chunks.jsonl           ──► BM25 index and source-of-truth chunks
   └──► index manifest v2      ──► chunk files, IDs, settings fingerprint
```

Indexing is incremental and document-aware:

- an ingestion manifest keyed by SHA-256 prevents re-parsing unchanged files;
- re-indexing a changed document replaces its stale chunks instead of orphaning
  them in Chroma;
- a v2 index manifest records chunk IDs, source files, and the settings
  fingerprint;
- the API refuses to answer when the index was built with incompatible embedding
  or chunking settings (`INDEX_FINGERPRINT_MODE=enforce`).

---

## Request lifecycle

For `POST /chat` and `POST /chat/stream`:

1. **Rate limit** — a thread-safe token bucket checks the client IP
   (`RATE_LIMIT_PER_MINUTE`, default 60/min).
2. **Input guardrails** — empty/oversized queries and obvious prompt-injection
   phrases are rejected; optional PII detection is available.
3. **Index fingerprint check** — a mismatched index returns `409` unless the mode
   is `warn`.
4. **Conversation ID** — supplied IDs are validated; missing IDs become UUIDs.
5. **Graph invocation** — the query enters the LangGraph control loop under a
   checkpointed `thread_id`.
6. **Retrieval** — hybrid search returns candidates, the reranker selects the
   best evidence, and context is token-budgeted.
7. **Generation** — the LLM answers from citation-labelled context only.
8. **Grounding check** — a judge verifies support; one retry is allowed.
9. **Output guardrails** — ungrounded answers are rejected; missing citations
   become warnings rather than hard failures.
10. **Response** — answer, citations, sources, warnings, verdict, latency, and
    request/thread IDs are returned.

Streaming follows the same path, but emits `meta`, `progress`, `done`, or `error`
SSE events as graph nodes complete.

---

## Project layout

```text
api/              FastAPI app, routes, schemas, security, shared state
  static/         zero-build web UI (HTML, CSS, ES modules)
chunking/         structure-aware and recursive token chunkers
config/           typed settings, model profiles, cache bootstrap, CLI
data/             source corpus (gitignored)
evaluation/       golden set, deterministic metrics, runner, generator
generation/       provider-specific LLM construction
graph/            LangGraph state, nodes, prompts, checkpointer, builder
ingestion/        discovery, parsers, hashing, manifests, metadata
observability/    guardrails, health checks, JSONL traces
reranking/        local cross-encoder reranker
retrieval/        embeddings, Chroma/Qdrant, BM25, RRF, caches, indexing
tests/            unit, integration, API, and evaluation tests
var/              runtime artifacts: Chroma, indexes, traces, checkpoints
```

---

## Quickstart

### Prerequisites

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Optional: Ollama or a vLLM endpoint for fully local generation
- Optional: a Groq API key for hosted generation

### 1. Install

```bash
uv sync \
  --extra dev \
  --extra orchestration \
  --extra ingest \
  --extra retrieval \
  --extra generation \
  --extra api
```

`--all-extras` also works, but it installs the optional Langfuse and
ragnarok-ai integrations as well. The core evaluation path needs no extra
packages.

### 2. Configure

```bash
cp .env.example .env
```

For a CPU-only machine, the default `RESOURCE_PROFILE=cpu` is appropriate. The
example environment uses local Ollama so no cloud key is needed:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434
# LLM_MODEL is optional; the cpu profile supplies qwen3:4b.
```

Pull the local model once:

```bash
ollama pull qwen3:4b
```

To use hosted Groq instead:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
LLM_API_KEY=your-key
```

Inspect and validate the configuration:

```bash
uv run python -m config
uv run python -m config --check-dirs --show-cache-env
```

### 3. Ingest and index the corpus

```bash
uv run python -m ingestion --path data
```

This parses supported files, chunks them, indexes new chunks into Chroma and
BM25, writes `var/index/chunks.jsonl`, and updates the index manifest. Re-running
it skips unchanged files; pass `--force` to re-ingest.

### 4. Regenerate the golden set (after corpus changes)

```bash
uv run python -m evaluation.build_golden_set
```

The generator reads section headings from `var/index/chunks.jsonl` and fails if
a curated question references a missing section.

### 5. Run the evaluation

```bash
uv run python -m evaluation
```

Reports are written to `var/eval/evaluation-*.json`. Use
`--no-fail-on-regression` while calibrating.

### 6. Start the API and UI

```bash
uv run python -m api
```

Open:

```text
http://127.0.0.1:8000/        chat UI
http://127.0.0.1:8000/docs    OpenAPI documentation
http://127.0.0.1:8000/health  health report
```

The first query can be slow while local embedding and reranker models load. Set
`WARMUP_ON_STARTUP=true` (the default) to move that cost into startup.

### 7. Run the tests

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

---

## Configuration

Every setting is read from an environment variable with the same name in
upper-case (`embedding_model` → `EMBEDDING_MODEL`). Relative paths resolve
against the repository root, not the process working directory.

### Configuration groups

| Group | Key settings | Purpose |
|---|---|---|
| Application | `APP_NAME`, `ENVIRONMENT`, `RESOURCE_PROFILE`, `PROJECT_ROOT` | Identity and path anchor |
| Ingestion | `PDF_PARSER`, `CHUNK_STRATEGY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_CHARS` | Parse and chunk behavior |
| Embeddings | `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_MAX_SEQ_LENGTH` | Local vector model |
| Vector store | `VECTOR_STORE_BACKEND`, `VECTOR_STORE_COLLECTION`, `CHROMA_PERSIST_DIR`, `QDRANT_URL` | Dense persistence |
| Retrieval | `TOP_K`, `DENSE_CANDIDATES`, `SPARSE_CANDIDATES`, `RRF_K`, `DENSE_WEIGHT`, `SPARSE_WEIGHT` | Hybrid search |
| Reranking | `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANK_TOP_N`, `RETRIEVAL_MIN_RERANK_SCORE` | Cross-encoder filtering |
| Grading | `GRADER_MODE=reranker|llm` | Fast local grading or LLM-as-judge |
| Performance | `RETRIEVAL_CACHE_SIZE`, `WARMUP_ON_STARTUP`, `WARMUP_QUERY` | Latency controls |
| LLM | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MAX_TOKENS` | Generation backend |
| Conversation | `CHECKPOINT_BACKEND`, `CHECKPOINT_DB_PATH`, `CHECKPOINT_TTL_SECONDS`, `MAX_QUERY_REWRITES`, `MAX_GROUNDING_RETRIES` | State and control loop |
| Evaluation | `GOLDEN_SET_PATH`, `EVAL_*_MIN`, `EVAL_FAIL_ON_REGRESSION` | Quality gates |
| Observability | `LOG_LEVEL`, `JSON_LOGS`, `TRACE_DIR`, `LANGFUSE_*` | Logs and traces |
| Guardrails | `MAX_QUERY_CHARS`, `PII_DETECTION_ENABLED`, `RATE_LIMIT_PER_MINUTE` | Input/output protection |
| API | `API_HOST`, `API_PORT`, `API_WORKERS`, `ADMIN_API_TOKEN`, `CORS_ORIGINS`, `MAX_UPLOAD_MB` | Serving and auth |

### Resource profiles

Profiles supply defaults only. An explicit model variable always wins.

| Profile | Embedding | Reranker | Local LLM default |
|---|---|---|---|
| `standard` | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | `qwen3:8b` |
| `cpu` | `BAAI/bge-small-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `qwen3:4b` |

When `LLM_PROVIDER=groq`, the default model is `qwen/qwen3.8-27b` regardless of
profile.

### Validation rules worth knowing

The settings validator rejects configurations that would silently degrade
quality:

- `CHUNK_OVERLAP < CHUNK_SIZE`
- `EMBEDDING_MAX_SEQ_LENGTH >= CHUNK_SIZE` (avoids silent truncation)
- `TOP_K <= RERANK_TOP_N <= DENSE_CANDIDATES` and `SPARSE_CANDIDATES`
- at least one retrieval weight greater than zero
- `LLM_MAX_TOKENS < LLM_NUM_CTX`
- Qdrant selected without a URL or local path
- `API_WORKERS > 1` with memory or SQLite checkpoints

---

## How the first step was built: skeleton and typed config

The first implementation step established the boundaries that every later step
depends on. It deliberately did not add models or business logic.

### 1. Package skeleton

The repository was split by responsibility rather than by framework layer:

```text
config/        settings, profiles, cache bootstrap
ingestion/     parsing, hashing, manifests
chunking/      token-aware chunkers
retrieval/     embeddings, stores, BM25, fusion, reranking
graph/         state, prompts, nodes, checkpointer
generation/    LLM providers
evaluation/    golden set, metrics, runner
api/           HTTP layer
observability/ logs, traces, guardrails
```

Each directory is an importable package with a small `__init__.py` that exports
its public surface. The dependency direction is one-way: `api` depends on
`graph`, `graph` depends on `retrieval` and `generation`, and `config` depends on
nothing except Pydantic.

### 2. Typed settings as the single source of truth

`config/settings.py` defines one `Settings(BaseSettings)` model. The core
pattern is:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["local", "staging", "production", "test"] = "local"
    resource_profile: Literal["standard", "cpu"] = "standard"
    project_root: Path = PROJECT_ROOT
    embedding_model: str = ""
    chunk_size: int = Field(default=512, ge=64, le=8192)
    llm_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _apply_profile_and_validate(self) -> "Settings": ...
```

Three details make this more than a `.env` wrapper:

**Project-root discovery.** `PROJECT_ROOT` is found by walking up to
`pyproject.toml`, or set explicitly. Every path field is then resolved against
that root. Starting the API from another directory cannot create a second empty
Chroma database.

**Cross-field validation.** Checks such as `TOP_K <= RERANK_TOP_N` prevent a
configuration that looks valid field-by-field but fails logically. Errors surface
at startup, not during a user request.

**Index fingerprint.** `Settings.index_fingerprint` hashes every setting that
changes vector semantics: embedding model, passage prefix, normalization, max
sequence length, chunk strategy/size/overlap, vector backend, and collection.
Query-only settings are excluded because they need a cache flush, not a full
reindex. The index manifest stores this fingerprint, and the API refuses to
serve a mismatched index in `enforce` mode.

### 3. Model profiles

`config/presets.py` defines coherent model sets for `standard` and `cpu`
hardware. The validator fills blank model fields from the selected profile, but
explicit environment values always take priority. This avoids a silent model
switch after an index has already been built.

### 4. Configuration CLI

`python -m config` prints the validated settings with secrets redacted;
`--check-dirs` verifies writable runtime directories and `--show-cache-env`
prints the Hugging Face cache environment. This made configuration debuggable
before any model existed in the repository.

### 5. Dependency strategy

Dependencies are split into optional extras (`orchestration`, `ingest`,
`retrieval`, `generation`, `api`, `observability`, `dev`). A developer can install
the base configuration layer without pulling Chroma, Docling, or a model runtime.
The lockfile pins the full environment so the API, ingestion CLI, and evaluation
harness resolve identically.

---

## Frontend and backend connection

The UI is intentionally a zero-build application:

```text
api/static/index.html   shell and semantic layout
api/static/styles.css   responsive dark theme
api/static/app.js       fetch, SSE parsing, DOM rendering, admin tools
```

FastAPI mounts the directory at `/static` and serves `index.html` at both `/`
and `/ui`. Because the UI is same-origin, no CORS configuration is needed and the
strict CSP can stay at `'self'`.

### Same-origin request paths

`app.js` uses relative URLs:

```js
fetch("/health")
fetch("/documents")
fetch("/chat/stream", { method: "POST", ... })
```

If you place the UI on a different origin, the supported patterns are:

1. put both behind one reverse proxy so they remain same-origin; or
2. set `CORS_ORIGINS=https://ui.example.com` and add an explicit API base URL in
   `app.js` (`fetch(`${API_BASE}/chat/stream`)`).

The code currently ships with relative paths because same-origin is the simplest
and safest default.

### Streaming without `EventSource`

The browser uses `fetch()` plus a `ReadableStream`, not the native
`EventSource` API. `EventSource` cannot send a JSON POST body, and the chat
request needs both. The client:

1. posts `{query, conversation_id}` to `/chat/stream`;
2. reads chunks with `TextDecoder`;
3. splits on blank lines into SSE blocks;
4. parses `event:` and `data:` lines;
5. updates the pending assistant message on `progress`;
6. renders the final answer, warnings, and sources on `done`.

If the stream emits `error`, the pending message is replaced with an error state
and the composer is re-enabled.

### Conversation identity

- `localStorage["lawhelp.conversationId"]` stores the active conversation UUID.
- “New chat” generates a fresh UUID and clears the transcript.
- The server validates IDs and also creates UUIDs when none is supplied.
- Conversation state is checkpointed by LangGraph under that `thread_id`.

There is no user account system. Conversation IDs isolate chats but are not an
authentication boundary: anyone who knows an ID could resume it. The supported
deployment is a trusted internal network behind a reverse proxy. Add real user
authentication before exposing conversations to untrusted users.

### Admin token

The UI keeps the admin token in `sessionStorage` only, and sends it as
`X-Admin-Token` for:

- `POST /ingest`
- `GET /jobs/{job_id}`
- `GET /documents`
- `DELETE /documents/{source_file}`

Chat endpoints do not require the admin token. In `local` and `test`
environments, an unset `ADMIN_API_TOKEN` leaves admin endpoints open for
developer convenience. In `staging` and `production`, an unset token denies
admin access.

### What the UI renders

- health pill with index status and chunk count;
- streaming progress labels for graph nodes;
- assistant answers with grounding verdict and latency;
- source cards with file, section, page, normalized match score, and snippet;
- warning banners for citation issues;
- knowledge-base list with chunk counts and delete buttons;
- admin upload, directory ingestion, and job polling controls.

---

## API reference

All endpoints are JSON unless noted. OpenAPI is available at `/docs` and
`/openapi.json`.

### `POST /chat`

Blocking request/response.

```bash
curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the standard partner rates?","conversation_id":"demo-1"}'
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Required, 1–100,000 characters before guardrails |
| `conversation_id` | string | Optional; validated or generated |
| `thread_id` | string | Backward-compatible alias for `conversation_id` |

Response:

```json
{
  "answer": "Partner rates range from $850 to $1,050 per hour [1].",
  "citations": ["rag_chatbot.txt | Section: 4.1 ..."],
  "sources": [
    {
      "id": 1,
      "source_file": "rag_chatbot.txt",
      "section": "4.1 Standard Hourly Rates by Attorney Title",
      "page": null,
      "chunk_id": "...",
      "score": 3.42,
      "snippet": "The firm's standard hourly rates ..."
    }
  ],
  "warnings": [],
  "retrieved_source_ids": ["..."],
  "grounding_verdict": "pass",
  "thread_id": "demo-1",
  "request_id": "5f...",
  "latency_ms": 1545.49,
  "trace_id": "demo-1"
}
```

### `POST /chat/stream`

Server-Sent Events. The response type is `text/event-stream`.

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is the Chicago office?","conversation_id":"demo-2"}'
```

Event sequence:

```text
event: meta
data: {"request_id":"...","thread_id":"demo-2"}

event: progress
data: {"node":"route","latency_ms":0.4}

event: progress
data: {"node":"retrieve","latency_ms":82.1}

...

event: done
data: {"answer":"...","sources":[...],"grounding_verdict":"pass", ...}
```

On failure the stream emits:

```text
event: error
data: {"code":"chat_failed","message":"The chat request failed.","request_id":"..."}
```

Graph node names in `progress` events: `route`, `condense_question`, `retrieve`,
`grade_documents`, `rerank`, `rewrite_query`, `generate`, `check_grounding`,
`direct_answer`, `log_trace`.

> **Why streaming is thread-bridged:** `SqliteSaver` implements only the sync
> checkpointer protocol. LangGraph's `astream()` calls async checkpoint methods
> and fails with `NotImplementedError`. The route therefore runs
> `graph.stream()` in a worker thread and forwards updates through a queue to the
> async response. Replacing this with `astream()` requires an async checkpointer
> such as `AsyncSqliteSaver` with `aiosqlite`.

### `POST /ingest`

Admin-only multipart endpoint. Send either a file or a server directory.

```bash
# Upload a file. The server streams it to disk with a hard size limit.
curl -X POST http://127.0.0.1:8000/ingest \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -F "file=@data/rag_chatbot.txt"

# Index a directory that already exists on the server.
curl -X POST http://127.0.0.1:8000/ingest \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -F "directory=data" \
  -F "force=false"
```

Response:

```json
{
  "status": "queued",
  "job_id": "9d...",
  "message": "Upload accepted and queued for indexing (rag_chatbot.txt)."
}
```

Uploads are sanitized, limited to supported extensions, streamed to a per-job
folder, and rejected above `MAX_UPLOAD_MB`.

### `GET /jobs/{job_id}`

Admin-only ingestion job status.

```json
{
  "job_id": "9d...",
  "kind": "ingest",
  "status": "succeeded",
  "created_at": 1762760000.0,
  "updated_at": 1762760004.2,
  "documents": 1,
  "chunks": 76,
  "message": "indexed 76 chunks; skipped 0",
  "error": null
}
```

Statuses: `queued`, `running`, `succeeded`, `failed`. Job records are in memory
and expire after one hour.

### `GET /documents`

Admin-only list of indexed sources and chunk counts.

```json
{
  "documents": [{"source_file": "rag_chatbot.txt", "chunks": 76}],
  "total_chunks": 76
}
```

### `DELETE /documents/{source_file}`

Admin-only. Removes a source from Chroma, BM25, `chunks.jsonl`, the index
manifest, and the ingestion manifest, then rebuilds the retriever and graph.

```bash
curl -X DELETE http://127.0.0.1:8000/documents/rag_chatbot.txt \
  -H "X-Admin-Token: $ADMIN_API_TOKEN"
```

### `GET /health`

Returns overall status plus config, vector store, BM25, ingestion, model cache,
and index fingerprint checks. The UI uses the index chunk count for its status
pill.

### `GET /metrics`

Process-local counters and cache statistics:

```json
{
  "request_count": 12,
  "chat_count": 5,
  "ingest_count": 1,
  "error_count": 0,
  "trace_count": 5,
  "last_trace_path": "var/traces/traces.jsonl",
  "caches": {
    "retrieval": {"entries": 2, "max_entries": 128, "hits": 3, "misses": 2},
    "reranker": {"entries": 2, "max_entries": 128, "hits": 3, "misses": 2}
  }
}
```

Counters reset on restart. They are intended for local operations, not as a
replacement for Prometheus.

### Error codes

| Status | Code | Meaning |
|---:|---|---|
| 400 | `invalid_input` | Guardrails rejected the query |
| 400 | `invalid_conversation_id` | ID contains unsupported characters |
| 400 | `unsupported_file_type` | Upload extension not supported |
| 401 | `admin_auth_required` | Missing or invalid admin token |
| 404 | `job_not_found` | Unknown or expired job |
| 404 | `document_not_found` | No chunks for that source |
| 409 | `index_fingerprint_mismatch` | Index built with other settings |
| 413 | `upload_too_large` | Upload exceeded `MAX_UPLOAD_MB` |
| 422 | `ungrounded_answer` | Answer had no supporting context |
| 429 | `rate_limit_exceeded` | Token bucket empty |
| 500 | `chat_failed` | Unhandled graph or provider failure |
| 503 | `application_not_initialized` | Startup failed or still running |

---

## Ingestion and indexing

### Supported formats

| Extension | Parser |
|---|---|
| `.txt` | UTF-8 (with BOM), UTF-8, then Latin-1 fallback |
| `.md`, `.markdown` | Text plus first heading detection |
| `.pdf` | pdfmux (default), Unstructured, or Docling |
| `.docx` | Unstructured or Docling |

### Chunking strategies

- **`structure` (default)** — heading-aware chunking that respects markdown
  headings, tables, and section boundaries. This corpus produces 76 chunks with
  section metadata such as `4.1 Standard Hourly Rates by Attorney Title`.
- **`recursive`** — token-aware recursive splitting for documents without
  reliable structure.

Both strategies use `tiktoken` for token counting and enforce
`CHUNK_OVERLAP < CHUNK_SIZE` and `MIN_CHUNK_CHARS` at the settings layer.

### Idempotency and replacement

Two manifests serve different jobs:

- `var/ingestion/manifest.json` maps file SHA-256 → ingestion metadata. It makes
  repeated ingestion of unchanged files cheap.
- `var/index/manifest.json` (schema v2) maps chunk IDs → metadata and source file
  → chunk IDs. It makes replacement and deletion possible.

When a document changes, the indexer removes that document's stale chunk IDs
from Chroma and rewrites the JSONL chunk store. It does not delete unrelated
chunks. The source-of-truth chunk list is always `var/index/chunks.jsonl`; Chroma
is a derived index.

### Upload safety

Uploads are:

- restricted to supported extensions;
- stripped to a safe basename (no traversal, no path separators);
- written to `var/uploads/<job_id>/<safe_name>`;
- streamed in 1 MiB blocks with an enforced byte limit;
- cleaned up on failure (`.part` files are removed).

---

## Retrieval pipeline

### Hybrid search

```text
query
  ├── dense: embedding → Chroma similarity search (DENSE_CANDIDATES)
  └── sparse: BM25 top-N (SPARSE_CANDIDATES)
                    │
                    ▼
          Reciprocal Rank Fusion (RRF_K, weights)
                    │
                    ▼
          cross-encoder rerank (RERANK_TOP_N)
                    │
                    ▼
          score threshold + token budget
```

RRF combines ranks rather than raw scores, so dense cosine similarity and BM25
scores do not need to share a scale. Fusion runs the two searches concurrently.
Ties are broken deterministically by document key.

### Grading modes

`GRADER_MODE` controls how irrelevant candidates are removed:

- **`reranker` (default)** — the local cross-encoder in the next node decides
  relevance. No LLM calls are spent on grading, which removes one call per
  retrieved document from the hot path.
- **`llm`** — an LLM judges each document (`GRADE_PROMPT`). This is useful for
  debugging and for tests, but it is much slower and more expensive. If the LLM
  judge returns no relevant documents, the graph falls back to the reranker
  before triggering a rewrite, reducing false-negative rewrites.

### Caching

Two thread-safe LRU caches reduce repeated work:

- **retrieval cache** keyed by normalized query;
- **reranker cache** keyed by query plus document identity.

Cached documents are copied before returning, so callers cannot mutate cached
metadata. Caches are rebuilt when new documents are indexed, so stale results do
not survive a reindex. `RETRIEVAL_CACHE_SIZE=0` disables caching.

Measured on the development CPU profile:

| Operation | Cold | Warm (cache hit) |
|---|---:|---:|
| Hybrid retrieval | 83.7 ms | 0.10 ms |
| Cross-encoder rerank | 458.8 ms | 0.11 ms |

### Warmup

`WARMUP_ON_STARTUP=true` runs a small query through the embedding model,
retriever, and reranker during application startup. Failures are logged but do
not prevent startup; the first real request surfaces the same error.

---

## Graph control loop

```text
             ┌─────────┐
             │  START  │
             └────┬────┘
                  ▼
              ┌───────┐
              │ route │
              └───┬───┘
        chit_chat / out_of_scope
                  │        └──────────────► direct_answer ──┐
                  ▼                                         │
          condense_question                                 │
                  ▼                                         │
              retrieve ──► grade_documents                  │
                              │                             │
                 documents?   │                             │
                    ├── yes ──► rerank ──► generate         │
                    └── no  ──► rewrite_query ──► retrieve │
                                                  │         │
                                                  ▼         │
                                            check_grounding │
                                              │      │      │
                                        retry│      │pass  │
                                              ▼      ▼      │
                                          generate  log_trace ◄┘
                                                       │
                                                      END
```

Key behaviors:

- **Heuristic routing first.** Firm-specific terms are routed to `rag` without
  an LLM call; greetings and clearly unrelated messages go to `direct_answer`.
- **Conversation condensation.** Follow-up questions are rewritten into
  standalone questions using the last six messages.
- **Corrective retrieval.** Empty retrieval triggers up to `MAX_QUERY_REWRITES`
  rewrites before a refusal.
- **Grounded generation.** The prompt includes numbered context blocks and
  instructs the model to cite every factual statement. When the exact answer is
  absent, it answers the closest supported question instead of saying “no such
  word” or “not in the provided context.”
- **Subjective questions.** Ranking questions such as “best attorneys” do not
  trigger a fabricated ranking. The assistant states that the documents do not
  rank attorneys, then offers factual differentiators such as practice
  leadership, years in practice, office, and representative matters. Subjective
  attorney queries are expanded to directory vocabulary before retrieval.
- **Pure refusals hide sources.** A refusal that answers nothing has no
  supporting citations, so unrelated retrieved chunks are not shown as sources.
- **Grounding retry.** A judge checks the answer; a failed check retries
  generation up to `MAX_GROUNDING_RETRIES`.
- **Citation parsing.** Bracketed labels are parsed defensively; only labels
  that map to retrieved sources become `sources`.
- **Context budgeting.** Chunks are selected under `CONTEXT_MAX_TOKENS` using
  token counts, not character guesses.

### Checkpointing

`CHECKPOINT_BACKEND=sqlite` (default) persists conversations in
`var/checkpoints.sqlite` across restarts. `memory` is available for tests and
ephemeral runs. `CHECKPOINT_TTL_SECONDS` tracks conversation activity and evicts
stale entries from the in-process activity map.

The SQLite checkpointer is synchronous. Do not switch the streaming route back to
`graph.astream()` without replacing it with an async checkpointer.

---

## Library choices and alternatives

| Concern | Chosen | Alternatives | Why this choice |
|---|---|---|---|
| API | **FastAPI** | Flask, Django, Litestar | Native async, Pydantic request/response validation, OpenAPI generation, `StreamingResponse` for SSE, and dependency injection in a small codebase. Flask needs more plumbing for async streaming; Django is heavier than this service warrants. |
| Config | **Pydantic Settings** | `python-dotenv` + `os.environ` | Typed fields, `SecretStr` redaction, `.env` loading, cross-field validators, and one validated object shared by CLI, API, ingestion, and eval. Plain env vars defer every error to runtime. |
| Orchestration | **LangGraph** | LangChain chains, custom loop, LlamaIndex | The control loop is a state machine with conditional edges, retries, and checkpointing. LangGraph models that explicitly and streams per-node updates, which the UI uses. A chain hides branching; a custom loop reimplements checkpointing. |
| Conversation state | **LangGraph `SqliteSaver`** | Redis, Postgres, in-memory dict | Persistent across restarts with zero services on a single node. A dict loses history; Redis/Postgres add operational burden that is only justified when scaling beyond one worker. |
| Vector store | **Chroma** | FAISS, pgvector, Qdrant, Pinecone | Persistent local collection, metadata filtering, simple Python API, and no server. FAISS lacks persistence ergonomics; pgvector requires Postgres; Pinecone is hosted. Qdrant is available when scale or filtering needs grow. |
| Sparse retrieval | **`rank_bm25`** | Elasticsearch, OpenSearch, Tantivy | Exact term matching for names, section numbers, and legal phrases with no service to run. Elasticsearch is superior at corpus scale but disproportionate here. The BM25 index is rebuilt from `chunks.jsonl` at startup. |
| Embeddings | **`sentence-transformers` / `bge-small-en-v1.5`** | OpenAI embeddings, Cohere, Voyage | Local inference, no per-query cost, no document text leaving the machine. `bge-m3` is available in the `standard` profile for multilingual and long-context needs. |
| Reranking | **Cross-encoder `ms-marco-MiniLM`** | LLM-as-judge, Cohere Rerank, no reranking | A cross-encoder scores query–document pairs directly and runs in milliseconds on CPU. LLM grading costs one call per document; hosted rerankers send data off-node. |
| Fusion | **RRF** | Weighted score normalization, learned fusion | Rank-based fusion is robust when dense and sparse scores have different scales, and it needs no training data. |
| LLM access | **Groq / Ollama / vLLM via LangChain** | Direct SDKs, OpenAI-only | One interface for hosted and local models. Groq is fast for hosted inference; Ollama is the simplest local option; vLLM provides an OpenAI-compatible high-throughput server. |
| Tokenization | **`tiktoken`** | Character estimates, model tokenizer downloads | Accurate context budgeting without loading a model tokenizer for every request. |
| Streaming | **SSE over `StreamingResponse`** | WebSockets, polling | The server only pushes events; there is no bidirectional channel to justify WebSockets. SSE works through ordinary HTTP proxies and is trivial to parse. |
| Frontend | **Zero-build HTML/CSS/ES modules** | React, Vue, Next.js, Svelte | No Node toolchain, no bundler, no framework upgrade cycle, and the entire UI is auditable in three files. The cost is manual DOM code; acceptable for a focused chat interface. |
| PDF parsing | **pdfmux** (default) | `pypdf`, Unstructured, Docling | Layout-aware extraction with page metadata and quality controls. `pypdf` is faster but loses structure; Docling is heavier and can trigger large model downloads. |
| Logging | **`structlog`** | stdlib `logging` only | Structured JSON events with request context, while remaining compatible with standard logging. |
| Evaluation | **Deterministic lexical metrics** | RAGAS, llamella NLI, LLM judges | No model download, no API cost, reproducible in CI, and enough to catch retrieval drift and unsupported answers. The tradeoff is that lexical overlap cannot judge legal nuance. |
| Testing | **pytest** | `unittest` | Fixtures, parametrization, monkeypatching, and fast failure output with less boilerplate. |
| Linting/formatting | **Ruff** | Black + Flake8 + isort | One fast tool for linting, import sorting, and formatting, with a single configuration. |

### Why `sentence-transformers==3.4.1` is pinned

The pin keeps the lockfile reproducible and matches the version range supported
by the current `langchain-huggingface` integration. Upgrading it should be done
deliberately: bump the pin, run the full test suite, and rebuild the index if the
embedding model changes. Model changes invalidate vectors even when the library
version does not.

---

## Security and guardrails

### Authentication and authorization

- Admin endpoints require `X-Admin-Token` or `Authorization: Bearer ...`.
- Tokens are compared with `secrets.compare_digest`.
- In `local`/`test`, an unset `ADMIN_API_TOKEN` leaves admin endpoints open. In
  `staging`/`production`, an unset token denies access.
- Chat endpoints are intentionally open for the internal UI; conversation IDs are
  isolation labels, not user authentication.

### HTTP hardening

`api/main.py` sets:

- `Content-Security-Policy` restricted to `'self'` (no inline scripts/styles);
- `X-Content-Type-Options: nosniff`;
- `X-Frame-Options: DENY`;
- `Referrer-Policy: no-referrer`;
- `Permissions-Policy` disabling camera, microphone, and geolocation.

CORS is disabled unless `CORS_ORIGINS` is set, and credentials are not allowed.

### Upload hardening

- supported extensions only;
- filename sanitization to a safe basename;
- per-job upload directories;
- streamed 1 MiB writes with a hard byte limit;
- partial-file cleanup on failure.

### Input guardrails

- query length limit (`MAX_QUERY_CHARS`);
- optional PII detection (SSN, credit card, phone, email);
- prompt-injection phrase detection.

These are heuristics, not a complete defense. Treat all document text and model
output as untrusted, and keep the service behind an authenticated reverse proxy
when it is exposed beyond localhost.

### Output guardrails

- citation presence is checked against retrieved sources;
- ungrounded answers are rejected with `422`;
- refusals are allowed when no context exists;
- missing citations produce warnings so the answer remains usable.

### Rate limiting

A thread-safe token bucket limits requests per client IP. It is in-process, so it
resets on restart and does not coordinate across workers. Put a shared limiter or
reverse-proxy limit in front of a multi-instance deployment.

---

## Evaluation

### Golden set

`evaluation/golden_set.jsonl` contains 26 curated questions spanning firm facts,
practice areas, attorney rates, billing, offices, engagement workflow,
compliance, glossary terms, and answering boundaries.

Each item uses a stable section key rather than a chunk ID:

```json
{
  "question": "What is Richard A. Bennett's standard hourly rate?",
  "expected_answer": "Richard A. Bennett bills $1,050 per hour.",
  "relevant_section_keys": [
    "rag_chatbot.txt::9.2 Canonical Rate Facts"
  ],
  "category": "attorney_rates"
}
```

Chunk IDs embed the document hash and chunk index, so they change when a file is
edited or the chunker is retuned. `source_file::section` survives re-chunking,
which makes the golden set maintainable.

Regenerate it from the current chunk store:

```bash
uv run python -m evaluation.build_golden_set
```

The generator validates that every referenced section exists and fails loudly if
the corpus no longer contains it.

### Metrics

The default scorer is deterministic and needs no model downloads:

| Metric | Definition |
|---|---|
| `groundedness` | Blend of answer-token coverage by retrieved context and sentence-level support |
| `completeness` | Fraction of expected-answer content tokens present in the answer |
| `relevance` | Query-token coverage in the answer and retrieved context |
| `consistency` | Grounding proxy with a small penalty when no citation is shown |
| `confidence` | Weighted blend of groundedness, completeness, and relevance |
| `iqs` | Mean of groundedness, completeness, relevance, and consistency |
| `retrieval_precision` | Relevant retrieved sections / retrieved sections |
| `retrieval_recall` | Relevant retrieved sections / all relevant sections |
| `reciprocal_rank` | 1 / rank of the first relevant section |
| `hit_rate` | Whether any relevant section appears in top-k |

Precision is naturally low when each question names one relevant section and the
retriever returns six chunks. The quality gate therefore uses **hit rate** and
**MRR**, with a low precision floor to catch retrieval that returns nothing
useful.

### Thresholds

Defaults in `.env.example`:

```dotenv
EVAL_GROUNDEDNESS_MIN=0.85
EVAL_COMPLETENESS_MIN=0.75
EVAL_RELEVANCE_MIN=0.75
EVAL_CONSISTENCY_MIN=0.80
EVAL_CONFIDENCE_MIN=0.70
EVAL_IQS_MIN=0.80
EVAL_RETRIEVAL_PRECISION_MIN=0.15
EVAL_RETRIEVAL_HIT_RATE_MIN=0.95
EVAL_RETRIEVAL_MRR_MIN=0.85
EVAL_FAIL_ON_REGRESSION=true
```

### Current baseline

Measured on the bundled synthetic corpus (26 questions, CPU profile, Groq
generation):

| Metric | Value |
|---|---:|
| Groundedness | 0.979 |
| Completeness | 0.867 |
| Relevance | 0.904 |
| Consistency | 0.979 |
| Confidence | 0.930 |
| IQS | 0.932 |
| Retrieval precision | 0.191 |
| Retrieval recall | 1.000 |
| MRR | 0.942 |
| Hit rate | 1.000 |

Reports are written to `var/eval/evaluation-<timestamp>.json` with per-item
answers, sources, retrieval metrics, and quality scores.

### Limitations of lexical evaluation

Token overlap cannot detect a subtly false legal statement, a paraphrase that
changes meaning, or a citation that points to the wrong sentence. Use the
deterministic metrics for regression detection, and add human review or an
optional LLM-judge for release decisions. The harness deliberately avoids a
large NLI model so evaluation stays fast and offline-friendly.

---

## Operations

### Single-node assumptions

The default deployment is one process because:

- SQLite checkpoints are local;
- BM25 is an in-memory index rebuilt at startup;
- rate limiting, jobs, counters, and caches are process-local;
- the index lock protects reindexing within one process.

The settings validator rejects `API_WORKERS > 1` with memory or SQLite
checkpoints. To scale out, move checkpoints to a shared store, move rate limiting
and jobs to Redis, and use a shared vector/search backend.

### Persistent state to back up

```text
var/chroma/             vector collection
var/index/              chunks.jsonl + manifest.json
var/ingestion/          ingestion manifest
var/checkpoints.sqlite  conversations
var/uploads/            uploaded source files
var/traces/             run traces
```

Losing `var/index/chunks.jsonl` means the BM25 index and source list must be
rebuilt by re-ingesting the corpus. Losing `var/chroma` requires reindexing.
Losing `var/ingestion` causes unchanged files to be parsed again.

### Reindex after a model or chunking change

1. Update the setting.
2. Run ingestion with `--force` (CLI) or `force=true` (API).
3. Confirm `/health` reports `index.status = ok` and the expected fingerprint.
4. Re-run the evaluation.

In `INDEX_FINGERPRINT_MODE=enforce`, chat returns `409` until the index is
rebuilt. In `warn`, it serves with a warning.

### Logs and traces

- Application logs are JSON (`JSON_LOGS=true`) with request method, path, status,
  and duration.
- Each graph run can append a `RunTrace` to `var/traces/traces.jsonl` with query,
  retrieved chunk IDs, reranker scores, route, grounding verdict, response, and
  node latencies.
- Langfuse is optional and disabled by default. Only self-hosted credentials are
  documented; no cloud credentials are required.

### Deployment checklist

- [ ] Set `ENVIRONMENT=production`.
- [ ] Set a strong `ADMIN_API_TOKEN`; never commit `.env`.
- [ ] Keep `API_WORKERS=1` unless you have an external checkpoint store.
- [ ] Bind to localhost and put TLS + authentication in a reverse proxy.
- [ ] Set `CORS_ORIGINS` only if the UI is on another origin.
- [ ] Cap request and upload sizes at the proxy as well as in the app.
- [ ] Back up `var/` state before upgrades.
- [ ] Monitor `/health` and `/metrics`; ship JSON logs to your log store.
- [ ] Run the evaluation before and after corpus/model changes.

No Docker, Kubernetes, Terraform, or CI configuration is included. The project is
intended to run directly from a pinned `uv` environment on a hardened node.

---

## Testing and quality gates

```bash
uv run pytest -q                 # full test suite
uv run ruff check .              # lint
uv run ruff format --check .     # formatting
```

The suite currently passes **121 tests** and covers:

- typed configuration and cross-field validation;
- ingestion, PDF metadata, chunking budgets, and idempotency;
- dense/BM25 retrieval, RRF fusion, reranking, caches, and index replacement;
- graph routing, correction, grounding retries, and multi-turn condensation;
- checkpointer behavior and streaming;
- API auth, rate limits, uploads, jobs, documents, and static assets;
- evaluation loading, scoring, thresholds, and golden-set generation.

Use the `Makefile` where GNU Make is available:

```bash
make test
make lint
make check
```

On Windows without GNU Make, run the equivalent `uv run ...` commands directly.

---

## Troubleshooting

### `409 index_fingerprint_mismatch`

The index was built with different embedding, chunking, or vector-store settings.
Re-run ingestion with force so the index is rebuilt, then retry.

```bash
uv run python -m ingestion --path data --force
```

### `NotImplementedError: The SqliteSaver does not support async methods`

This occurs if streaming is switched back to `graph.astream()` while using the
sync `SqliteSaver`. The current route avoids it by running `graph.stream()` in a
worker thread. To use async streaming, install `aiosqlite` and switch to
`AsyncSqliteSaver`, then verify shutdown and checkpoint behavior.

### First request is slow

Local models load on first use. Keep `WARMUP_ON_STARTUP=true`, or warm the cache
manually by sending a benign query after startup. Later identical queries hit the
retrieval and rerank caches.

### `401 admin_auth_required`

Set `ADMIN_API_TOKEN` and send it as `X-Admin-Token`. In `local`, an unset token
allows admin calls; in `staging`/`production`, it does not.

### No documents retrieved

Check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/documents -H "X-Admin-Token: $ADMIN_API_TOKEN"
```

If chunk counts are zero, re-run ingestion. If the fingerprint is mismatched,
force a reindex. If only BM25 is empty, check `var/index/chunks.jsonl`.

### `422 ungrounded_answer`

The model produced an answer with no supporting context. Check retrieval, the
generation prompt, and the grounding judge. For chit-chat or out-of-scope routes,
the response should be a direct answer rather than retrieval-based.

### Model download fails or the machine is offline

Pre-populate `.cache/huggingface` on a connected machine and copy it over, or set
`HF_HUB_OFFLINE=1` after the cache is populated. The runtime does not require a
network connection when the cache is complete.

### Windows terminal encoding errors

The corpus contains an em dash. PowerShell may display it as `�?"` depending on
the code page; the files are UTF-8 and the Python parsers read them correctly.
Set `$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()` if
you need readable console output.

---

## Current limitations

- **Single-node by design.** In-memory jobs, counters, rate limits, and caches do
  not coordinate across workers.
- **No user accounts.** Conversation IDs isolate chats but are not authentication.
- **Heuristic guardrails.** Prompt-injection and PII detection are pattern-based
  and can be bypassed; they are a first line of defense, not a policy engine.
- **Lexical evaluation.** Deterministic metrics detect drift, not legal nuance.
- **BM25 is rebuilt in memory.** Startup cost grows with corpus size; very large
  corpora should use a dedicated search service.
- **Chroma is local.** Large collections and multi-tenant filtering may warrant
  Qdrant or another server-backed store.
- **Citation labels are not verification.** The model is instructed to cite, and
  the output guardrail checks for citations, but a human must verify legal
  accuracy before relying on an answer.
- **No container or CI files.** Deployment is a pinned Python environment on a
  hardened host, by design.

This assistant is a knowledge-retrieval tool, not legal advice. It does not create
an attorney-client relationship, and answers must be checked against the cited
source documents before use.
