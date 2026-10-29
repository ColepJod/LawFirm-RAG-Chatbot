# Law Firm RAG Chatbot — Codebase Overview

This document is a detailed, step-by-step summary of the repository so another LLM agent or engineer can understand, navigate, and extend the codebase without re-reading every file.

Repository: `https://github.com/ColepJod/LawFirm-RAG-Chatbot.git`

---

## 1. Project purpose

This is a **local-first, multi-document Retrieval-Augmented Generation (RAG) chatbot** for a law firm. It is designed for an arbitrary corpus of PDF, Markdown, plain-text, and DOCX documents, not a single PDF.

Core goals:

- answer knowledge questions from firm documents,
- preserve citations and grounding,
- support multi-turn conversations,
- evaluate retrieval and answer quality locally,
- expose the system through a FastAPI API,
- remain free/open-source and runnable locally or in staging,
- avoid mandatory paid/cloud APIs.

The default LLM provider is now **Groq**, but Ollama and vLLM remain supported. Embeddings, reranking, and evaluation models run locally from Hugging Face.

---

## 2. High-level architecture

```text
User query
  -> FastAPI /chat
  -> input guardrails
  -> LangGraph RAGState
  -> route
     -> direct_answer (chit-chat / out-of-scope)
     -> condense_question
     -> retrieve (dense + BM25 -> RRF)
     -> grade_documents
        -> rewrite_query -> retrieve (bounded retries)
        -> rerank
     -> generate
     -> check_grounding
        -> generate (bounded retries)
     -> log_trace
  -> output guardrails
  -> JSON response
```

Supporting systems:

- ingestion: parse and hash documents
- chunking: recursive or structure-aware
- indexing: Chroma/Qdrant + BM25
- reranking: local cross-encoder
- evaluation: llamella + retrieval metrics
- observability: JSON traces, health checks, guardrails

---

## 3. Technology stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| Framework | LangChain |
| API | FastAPI |
| PDF parsing | pdfmux, Unstructured OSS, Docling |
| Vector store | ChromaDB default, Qdrant optional |
| Embeddings | Hugging Face sentence-transformers |
| Sparse retrieval | `rank_bm25` |
| Reranker | sentence-transformers `CrossEncoder` |
| LLM | Groq default, Ollama/vLLM optional |
| Evaluation | llamella, ragnarok-ai, optional RAGAS |
| Observability | Structured JSON logs, optional Langfuse |
| Testing | pytest, Ruff |

No Docker, Kubernetes, Terraform, CI/CD, or cloud deployment files are included.

---

## 4. Directory structure

```text
.
├── api/                 FastAPI application
├── chunking/            Chunking strategies
├── config/              Typed settings and model profiles
├── data/                Source documents (gitignored contents)
├── evaluation/          Golden set and evaluation harness
├── generation/          LLM construction
├── graph/               LangGraph state, nodes, builder
├── ingestion/           Multi-format document ingestion
├── observability/       Tracing, guardrails, health checks
├── reranking/           Local cross-encoder reranking
├── retrieval/           Embeddings, vector stores, BM25, fusion, indexing
├── tests/               Unit and integration tests
├── var/                 Runtime data, indexes, logs, traces (gitignored contents)
├── .env.example         Configuration template
├── pyproject.toml       Dependencies and tool config
├── uv.lock              Universal dependency lockfile
├── Makefile             Local development tasks
└── README.md            Project documentation
```

---

## 5. Configuration

Configuration is centralized in `config/settings.py`.

Key concepts:

- `Settings` extends Pydantic `BaseSettings`.
- Environment variables map directly to field names, e.g. `EMBEDDING_MODEL -> embedding_model`.
- `.env` is loaded from the project root.
- Relative paths resolve against `PROJECT_ROOT`.
- `RESOURCE_PROFILE` supplies defaults only when a model field is empty.
- Explicit environment variables always override profile defaults.

Important settings:

| Setting | Purpose |
|---|---|
| `RESOURCE_PROFILE` | `standard` or `cpu` |
| `EMBEDDING_MODEL` | Local Hugging Face embedding model |
| `RERANKER_MODEL` | Local cross-encoder |
| `LLM_PROVIDER` | `groq`, `ollama`, or `vllm` |
| `LLM_MODEL` | LLM model name |
| `LLM_API_KEY` | Required for Groq |
| `VECTOR_STORE_BACKEND` | `chroma` or `qdrant` |
| `CHUNK_SIZE` | Token budget per chunk |
| `CHUNK_OVERLAP` | Overlap tokens |
| `TOP_K` | Final retrieved chunks |
| `DENSE_CANDIDATES` | Dense retrieval candidate depth |
| `SPARSE_CANDIDATES` | BM25 candidate depth |
| `RRF_K` | Reciprocal Rank Fusion constant |
| `MAX_QUERY_REWRITES` | Corrective RAG retry bound |
| `MAX_GROUNDING_RETRIES` | Grounding retry bound |
| `LOCAL_MODEL_CACHE_DIR` | Hugging Face cache directory |

Model profiles:

| Profile | Embedding | Reranker | Notes |
|---|---|---|---|
| `standard` | `BAAI/bge-m3` | `BAAI/bge-reranker-v2-m3` | GPU recommended |
| `cpu` | `BAAI/bge-small-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No GPU required |

---

## 6. Step-by-step implementation summary

### Step 1 — Project skeleton and configuration

Files:

- `pyproject.toml`
- `uv.lock`
- `.env.example`
- `config/settings.py`
- `config/presets.py`
- `config/cache.py`
- `config/__main__.py`
- `tests/test_config.py`

Implemented:

- project metadata and pinned dependencies,
- typed settings with validation,
- model profiles,
- Hugging Face cache bootstrap,
- path resolution,
- config CLI,
- configuration tests.

### Step 2 — Multi-format ingestion

Files:

- `ingestion/discovery.py`
- `ingestion/hashing.py`
- `ingestion/metadata.py`
- `ingestion/manifest.py`
- `ingestion/parsers.py`
- `ingestion/pipeline.py`
- `ingestion/__main__.py`
- `tests/test_ingestion.py`

Implemented:

- support for PDF, Markdown, TXT, DOCX,
- pdfmux primary PDF parser with metadata mapping,
- Unstructured and Docling alternatives,
- required metadata on every document,
- SHA-256 deduplication,
- idempotent ingestion manifest,
- CLI: `python -m ingestion`.

### Step 3 — Chunking layer

Files:

- `chunking/tokens.py`
- `chunking/context.py`
- `chunking/tables.py`
- `chunking/recursive.py`
- `chunking/structure.py`
- `chunking/pipeline.py`
- `tests/test_chunking.py`

Implemented:

- common chunker interface,
- recursive token chunking,
- structure-aware Markdown heading chunking,
- table-row protection,
- chunk context headers,
- token budget enforcement,
- chunk metadata.

### Step 4 — Embedding, indexing, and retrievers

Files:

- `retrieval/embeddings.py`
- `retrieval/vectorstores.py`
- `retrieval/bm25.py`
- `retrieval/indexing.py`
- `retrieval/retrievers.py`
- `tests/test_retrieval.py`

Implemented:

- lazy local embeddings,
- Chroma and Qdrant backends,
- BM25 index,
- incremental indexing,
- dense and sparse LangChain retrievers.

### Step 5 — Hybrid retrieval with RRF

Files:

- `retrieval/fusion.py`
- `retrieval/hybrid.py`
- `tests/test_fusion.py`

Implemented:

- explicit RRF fusion function,
- deduplication by chunk ID,
- `rrf_score` metadata,
- `HybridRRFRetriever`,
- configurable candidate depths and weights.

### Step 6 — Local reranking

Files:

- `reranking/cross_encoder.py`
- `reranking/pipeline.py`
- `tests/test_reranking.py`

Implemented:

- LangChain-compatible `CrossEncoderReranker`,
- local model loading,
- device resolution,
- top-N reranking,
- reranker score logging in metadata,
- disabled-reranker fallback.

### Step 7 — LangGraph RAG control loop

Files:

- `graph/state.py`
- `graph/prompts.py`
- `graph/nodes.py`
- `graph/builder.py`
- `tests/test_graph.py`

Implemented nodes:

- `route`
- `condense_question`
- `retrieve`
- `grade_documents`
- `rerank`
- `rewrite_query`
- `generate`
- `check_grounding`
- `direct_answer`
- `log_trace`

Implemented features:

- typed `RAGState`,
- dependency injection,
- conditional edges,
- corrective RAG rewrite loop,
- grounding retry loop,
- bounded retries,
- citation generation,
- refusal when context is insufficient.

### Step 8 — Multi-turn memory

Files:

- `graph/checkpointing.py`
- `graph/state.py` updates
- `graph/nodes.py` updates
- `tests/test_graph.py` multi-turn test

Implemented:

- MemorySaver by default,
- optional SQLite checkpointing,
- thread-level state via `thread_id`,
- condensation using checkpointed history,
- `retrieved_source_ids` tracking.

### Step 9 — Evaluation harness

Files:

- `evaluation/golden_set.py`
- `evaluation/metrics.py`
- `evaluation/llamella_scorer.py`
- `evaluation/thresholds.py`
- `evaluation/runner.py`
- `evaluation/ragnarok_adapter.py`
- `evaluation/__main__.py`
- `evaluation/golden_set.jsonl`
- `tests/test_evaluation.py`

Implemented:

- JSONL golden set,
- llamella deterministic scoring,
- retrieval metrics,
- threshold checks,
- JSON reports,
- CI-friendly exit codes,
- optional ragnarok-ai adapter.

### Step 10 — Observability and guardrails

Files:

- `observability/tracing.py`
- `observability/guardrails.py`
- `observability/health.py`
- `observability/__init__.py`
- `graph/builder.py` timed nodes and `log_trace`
- `tests/test_observability.py`

Implemented:

- structured JSON traces,
- per-node latency,
- input guardrails,
- output citation/refusal guardrails,
- health checks.

### Step 11 — FastAPI layer

Files:

- `api/main.py`
- `api/state.py`
- `api/factory.py`
- `api/schemas.py`
- `api/routes.py`
- `api/__main__.py`
- `tests/test_api.py`

Implemented endpoints:

- `POST /ingest`
- `POST /chat`
- `GET /health`
- `GET /metrics`

Features:

- compiled graph reused across requests,
- background ingestion,
- input/output guardrails,
- thread IDs for conversation memory.

### Step 12 — End-to-end tests

Files:

- `tests/test_end_to_end.py`

Implemented:

- ingestion → chunking → indexing → retrieval pipeline test,
- expected source in top results,
- graph refusal without context.

---

## 7. Data flow

### Ingestion

```text
file path
  -> discover_documents
  -> sha256_file
  -> parse_file
  -> base_metadata
  -> ingest_file
  -> IngestionManifest
```

### Chunking

```text
Document
  -> RecursiveTokenChunker or StructureAwareChunker
  -> table-aware blocks
  -> context header
  -> chunks with metadata
```

### Indexing

```text
chunks
  -> chunk_id
  -> embedding model
  -> Chroma/Qdrant
  -> BM25 index
  -> chunk store JSONL
  -> index manifest
```

### Retrieval

```text
query
  -> dense retriever
  -> BM25 retriever
  -> rrf_fuse
  -> CrossEncoderReranker
  -> top-N chunks
```

### Generation

```text
standalone query + reranked chunks
  -> generate
  -> check_grounding
  -> final answer + citations
```

---

## 8. LangGraph state

`RAGState` is a `TypedDict` with `total=False`.

Key fields:

- `messages`
- `original_query`
- `standalone_query`
- `retrieved_chunks`
- `retrieved_source_ids`
- `reranked_chunks`
- `generation`
- `grounding_verdict`
- `citations`
- `route_decision`
- `rewrite_count`
- `grounding_retry_count`
- `node_latencies`

`GraphDependencies` injects:

- `settings`
- `llm`
- `retriever`
- `reranker`
- optional override functions
- `trace_logger`

---

## 9. Key interfaces

### Chunker interface

```python
class Chunker(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]: ...
```

Implementations:

- `RecursiveTokenChunker`
- `StructureAwareChunker`

### Vector store handle

```python
VectorStoreHandle.backend_name
VectorStoreHandle.collection_name
VectorStoreHandle.store
VectorStoreHandle.add_documents(...)
VectorStoreHandle.as_retriever(k=...)
```

### Evaluation pipeline

```python
def pipeline(question: str) -> EvaluationOutput: ...
```

Returns:

- `answer`
- `contexts`
- `citations`
- `retrieved_source_ids`

---

## 10. Configuration and environment

Important environment variables:

```env
RESOURCE_PROFILE=cpu
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
LLM_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_STORE_BACKEND=chroma
TOP_K=6
MAX_QUERY_REWRITES=2
MAX_GROUNDING_RETRIES=1
LOCAL_MODEL_CACHE_DIR=.cache/huggingface
```

Paths are resolved relative to the project root, not the current working directory.

---

## 11. Testing

Test files:

- `tests/test_config.py`
- `tests/test_ingestion.py`
- `tests/test_chunking.py`
- `tests/test_retrieval.py`
- `tests/test_fusion.py`
- `tests/test_reranking.py`
- `tests/test_graph.py`
- `tests/test_evaluation.py`
- `tests/test_observability.py`
- `tests/test_api.py`
- `tests/test_end_to_end.py`

Run:

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Latest verified result:

```text
81 passed, 1 warning
```

---

## 12. Running the application

Install:

```powershell
uv sync --extra dev --extra ingest --extra orchestration --extra retrieval --extra evaluation --extra api --extra generation
```

Configure:

```powershell
Copy-Item .env.example .env
```

Set:

```env
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
LLM_API_KEY=your_key_here
```

Ingest:

```powershell
uv run python -m ingestion --path data
```

Serve:

```powershell
uv run python -m api
```

Evaluate:

```powershell
uv run python -m evaluation
```

---

## 13. Known limitations and risks

- Groq requires an API key and network access.
- `langchain-community` is being sunset; BM25 retriever may need migration.
- `llamella` has limited maintenance visibility and its GitHub URL returned 404.
- `sentence-transformers==3.4.1` is pinned for `llamella` compatibility, not because it is the newest release.
- The `transformers==4.57.6` and `sentence-transformers==3.4.1` combination was not runtime-tested for all embedding/reranker models.
- The golden set contains sample questions; users should replace or expand it for their corpus.
- No frontend UI is included; the system is API-first.
- No production deployment files are included by design.

---

## 14. Extension points

To add a new parser:

1. Implement a function in `ingestion/parsers.py`.
2. Return `list[Document]` with required metadata.
3. Add the file extension to `SUPPORTED_EXTENSIONS`.
4. Add tests.

To add a new chunker:

1. Implement `split_documents()` in `chunking/`.
2. Register it in `chunking/pipeline.py`.
3. Add tests.

To add a new LLM provider:

1. Add the provider to `LLM_PROVIDER`.
2. Implement construction in `generation/llm.py`.
3. Update `.env.example` and tests.

To add a new vector store:

1. Implement a backend in `retrieval/vectorstores.py`.
2. Add factory branch.
3. Add tests.

To add a new graph node:

1. Implement the node in `graph/nodes.py`.
2. Add it in `graph/builder.py`.
3. Add conditional edges if needed.
4. Add tests.

---

## 15. Final note

The project is complete through all 12 planned steps. The codebase is modular, tested, and runnable locally. The architecture intentionally separates ingestion, chunking, retrieval, reranking, generation, evaluation, observability, and API concerns so each can be changed independently.