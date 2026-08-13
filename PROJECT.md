# Project Document: PDF-based RAG Application

| Field | Value |
|-------|-------|
| **Project name** | PDF-based RAG Application |
| **Version** | 0.8.0 |
| **Type** | Retrieval-Augmented Generation (RAG) API |
| **Stack** | FastAPI · LangGraph · OpenAI · Qdrant · BM25 |
| **Primary interface** | `POST /ask` (also CLI + OpenAPI docs) |

---

## 1. Executive summary

This project is a **PDF question-answering system**. Users ingest one or more PDFs; the system chunks and embeds the text, stores vectors (and a keyword index), then answers natural-language questions using only retrieved document context. Every answer includes **page-level source citations** so responses stay grounded and auditable.

It is built as a clear, phased RAG pipeline: required ingest + retrieve + generate first, then hybrid search, reranking, pruning, query rewrite, observability, and evaluation.

---

## 2. Problem statement

Large language models alone often:

- Hallucinate facts not present in training data  
- Cannot cite a specific PDF page or section  
- Miss exact terms (model names, acronyms, URLs) when using only dense vectors  

**Goal:** Given a PDF corpus, answer user questions accurately, with citations, using a controllable retrieval → generation workflow rather than “paste the whole PDF into a prompt.”

---

## 3. Objectives

| Objective | Success criteria |
|-----------|------------------|
| Ingest PDFs | Load, chunk, embed, and store with metadata (`page`, `section`, `chunk_id`) |
| Answer questions | `POST /ask` returns grounded `answer` + `sources[]` + `latency_ms` |
| Improve retrieval | Hybrid dense + BM25 search, fused with RRF |
| Improve context quality | Cross-encoder rerank → prune to a small, high-signal set |
| Recover from weak retrieval | Conditional query rewrite and limited generate retry |
| Observe & measure | LangSmith traces + gold-set eval (baseline vs advanced) |

---

## 4. Scope

### In scope

- PDF text extraction (PyMuPDF), including tables/figures when present  
- Chunking with overlap and section heuristics  
- OpenAI embeddings + generation  
- Local (or remote) Qdrant vector store  
- BM25 sparse index + Reciprocal Rank Fusion  
- LangGraph orchestration (nodes + conditional edges)  
- FastAPI HTTP API and CLI  
- LangSmith tracing and offline evaluation harness  

### Out of scope

- Multi-modal vision agents over page images  
- Web search / external tool calling  
- Redis / semantic response cache (prompt-prefix cache only)  
- Multi-tenant auth, rate limiting, or production hardening  

---

## 5. Solution overview

```text
PDF  →  Extract  →  Chunk + metadata  →  Embed  →  Qdrant
                                      ↘           ↗
                                        BM25 index

Question  →  FastAPI  →  LangGraph
              hybrid retrieve → RRF → rerank → prune
              → quality check (rewrite if weak)
              → generate (retry if refusal)
              → Answer + sources + latency
```

**Ingest path:** PDF → pages → chunks → embeddings → Qdrant upsert + BM25 corpus rebuild.  
**Ask path:** Question → hybrid top-20 → rerank top-10 → prune ~5 → LLM answer with citations.

---

## 6. Architecture

### 6.1 Components

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| API | FastAPI + Uvicorn | `GET /health`, `POST /ingest`, `POST /ask` |
| Orchestration | LangGraph `StateGraph` | Wire retrieve → rerank → prune → quality → generate |
| PDF | PyMuPDF | Text / table / figure extraction |
| Chunking | Custom splitter | ~800 chars, 150 overlap, section-aware metadata |
| Dense retrieval | OpenAI `text-embedding-3-small` + Qdrant | Semantic similarity |
| Sparse retrieval | BM25 (`rank-bm25`) | Exact-term matching |
| Fusion | RRF (`k=60`) | Merge dense + sparse rankings |
| Rerank | Cross-encoder MiniLM | Score `(query, chunk)` pairs |
| Generation | OpenAI `gpt-4.1-mini` | Grounded answer from pruned context |
| Rewrite | OpenAI `gpt-5.4-nano` | Reformulate query when context is weak |
| Observability | LangSmith | Per-node spans, tokens, prompt-cache reads |
| Eval | JSONL gold set + CLI/notebook | Hit rate, MRR, faithfulness, latency, etc. |

### 6.2 LangGraph flow (advanced)

```text
START
  → retrieve (hybrid + RRF)
  → rerank
  → prune
  → quality_check
       ├─ context weak + retries left → rewrite_query → retrieve
       └─ otherwise → generate
            ├─ weak answer + usable context → generate (once)
            └─ OK → END (answer + sources)
```

### 6.3 Design principles

1. **Required first, optional later** — core PDF → vectors → `/ask` before advanced features.  
2. **LangGraph as orchestrator** — nodes are plain Python; no LangChain RetrievalQA chains.  
3. **Ground every answer** — LLM sees only pruned chunks; responses expose document/page/section/chunk.  
4. **Dense + sparse** — meaning and exact terms both matter.  
5. **Narrow context before generate** — fewer tokens, less noise, stronger grounding.  
6. **Observe and measure** — traces + reproducible eval.

---

## 7. Features

### Required (core RAG)

| Feature | Implementation |
|---------|----------------|
| PDF loading & text extraction | `app/core/pdf_loader.py` |
| Text chunking | `app/core/chunking.py` |
| Embedding generation | `app/core/embeddings.py` |
| Vector storage & similarity search | `app/core/vectorstore.py` (Qdrant) |
| Answer generation from context | `generate_node` + `llm.py` / `prompts.py` |
| Source references | `sources[]` in API response |
| FastAPI `POST /ask` | `app/api/routes.py` |
| LangGraph pipeline | `app/graph/` |

### Optional (implemented)

| Feature | Notes |
|---------|--------|
| Hybrid search | Qdrant + BM25 → RRF |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Context pruning | Score / overlap / token budget |
| Query rewrite + retries | Conditional LangGraph edges |
| Prompt caching | Frozen system prefix (OpenAI provider cache) |
| LangSmith tracing | Parent `rag_ask` + node spans |
| Evaluation metrics | Baseline vs advanced benchmark |

---

## 8. API specification

Base URL (local): `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

### `GET /health`

Liveness check (no Qdrant / LLM calls).

```json
{ "status": "ok" }
```

### `POST /ingest`

Load PDF(s) → chunk → embed → Qdrant + BM25.

**Request (optional body)**

```json
{ "pdf_path": "data/source_pdfs/Document.pdf" }
```

Omit `pdf_path` (or send `{}`) to ingest all PDFs under `data/source_pdfs/`.

**Response (example)**

```json
{
  "message": "Ingested Document.pdf.",
  "documents": [
    {
      "document": "Document.pdf",
      "pages": 42,
      "chunks": 180,
      "tables": 3,
      "figures": 1
    }
  ]
}
```

### `POST /ask`

**Request**

```json
{ "question": "What is Retrieval-Augmented Generation?" }
```

**Response**

```json
{
  "answer": "Retrieval-Augmented Generation is a technique that combines information retrieval with text generation...",
  "sources": [
    {
      "document": "Document.pdf",
      "page": 1,
      "section": "Introduction",
      "chunk_id": "chunk_3",
      "score": 5.12,
      "content_type": "text"
    }
  ],
  "latency_ms": 1420
}
```

| Field | Meaning |
|-------|---------|
| `answer` | LLM answer grounded in pruned chunks |
| `sources` | Chunks sent to the LLM (~`PRUNE_K`) |
| `sources[].page` | PDF page citation |
| `sources[].score` | Cross-encoder relevance score |
| `latency_ms` | Graph wall time (ms) |

---

## 9. Model choices

| Job | Model | Rationale |
|-----|-------|-----------|
| Embeddings | `text-embedding-3-small` | Strong retrieval quality at low cost |
| Generation | `gpt-4.1-mini` | Grounded Q&A, good instruction following, efficient on long prompts |
| Query rewrite | `gpt-5.4-nano` | Light rewrite / helper steps |
| Rerank | MiniLM cross-encoder (local) | Fast pairwise scoring without an extra LLM call |

Configure via `.env`. After changing the embedding model, **re-ingest** (vector dimensions may change).

---

## 10. Project structure

```text
app/
  main.py              # FastAPI entry
  config.py            # Settings from .env
  cli.py               # ingest / ask / eval CLI
  api/routes.py        # HTTP endpoints
  models/schemas.py    # Request/response models
  core/                # PDF, chunk, embed, Qdrant, BM25, hybrid, rerank, prune, LLM
  graph/               # LangGraph state, nodes, pipeline
  eval/                # Tracing, dataset, evaluators, benchmark
data/
  source_pdfs/         # Input PDFs
  eval/rag_eval.jsonl  # Gold evaluation set
tests/
vectorstore_db/        # Local Qdrant + bm25_corpus.json (created at runtime)
.env.example
requirements.txt
README.md              # Developer setup & deep dive
PROJECT.md             # This document
```

---

## 11. Setup and run (summary)

1. Create a virtualenv and install dependencies: `pip install -r requirements.txt`  
2. Copy `.env.example` → `.env` and set `OPENAI_API_KEY`  
3. Place PDFs in `data/source_pdfs/`  
4. Start API: `uvicorn app.main:app --reload`  
5. `POST /ingest` then `POST /ask`  

**CLI alternatives**

```bash
python -m app.cli ingest
python -m app.cli ask "Your question here"
python -m app.cli eval
```

Default storage is **embedded Qdrant** on disk (`QDRANT_PATH=vectorstore_db`). Leave `QDRANT_URL` empty for local use.

---

## 12. Configuration highlights

| Variable | Default | Role |
|----------|---------|------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Chunking |
| `RETRIEVE_K` / `RERANK_K` / `PRUNE_K` | 20 / 10 / 5 | Retrieval funnel |
| `CONTEXT_SCORE_THRESHOLD` | 0.5 | Trigger rewrite when context is weak |
| `MAX_RETRIEVE_RETRIES` | 2 | Rewrite loop limit |
| `MAX_GENERATE_RETRIES` | 1 | Generate retry on weak/refusal answers |
| `LANGSMITH_API_KEY` | empty | Optional tracing |

---

## 13. Evaluation and observability

- **LangSmith:** When `LANGSMITH_API_KEY` is set, each `/ask` run records a parent trace and per-node spans (retrieve, rerank, prune, rewrite, generate).  
- **Eval:** A gold JSONL set (`data/eval/rag_eval.jsonl`) supports baseline (vector → generate) vs advanced (full graph) comparison on retrieval and answer-quality metrics (hit rate, recall, precision, MRR, faithfulness, relevance, latency).

---

## 14. Limitations

- Quality depends on PDF text extractability (scanned/image-only PDFs need OCR, not included).  
- Answers are limited to ingested content; no live web retrieval.  
- Reranker downloads a local sentence-transformers model on first use.  
- Prompt caching is OpenAI provider-side (frozen system prefix), not a full response cache.  
- Not production-hardened (auth, quotas, multi-tenant isolation).

---

## 15. Future improvements

- OCR pipeline for scanned PDFs  
- Multi-document filters in `/ask` (by filename / section)  
- Persistent BM25 / Qdrant sync strategies for large corpora  
- Optional stronger generate model for multi-hop questions  
- Auth and rate limiting for shared deployments  

---

## 16. Conclusion

The PDF-based RAG Application delivers a complete, citation-aware Q&A stack over PDF corpora: ingest → hybrid retrieve → rerank → prune → generate, orchestrated with LangGraph and exposed through a simple FastAPI surface. Optional rewrite, tracing, and evaluation make the system inspectable and measurable beyond a basic “LLM + PDF” demo.
