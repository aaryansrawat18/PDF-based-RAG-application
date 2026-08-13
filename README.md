# PDF-based RAG application

Current phase: **1 — Setup + Basic LangGraph RAG** (no API yet).

Pipeline: PDF → chunks → Qdrant → LangGraph (`retrieve → generate`) → answer with page sources.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY` (Gemini) or `OPENAI_API_KEY` plus `LLM_PROVIDER=openai`.

Qdrant runs embedded on disk in `vectorstore_db/` when `QDRANT_URL` is empty. No Docker required.

## Run

From the repo root:

```bash
python -m app.cli ingest
python -m app.cli ask "What is RAG?"
```

If `data/source_pdfs/` has no PDFs, ingest writes a sample `rag_intro.pdf`. Put your own PDFs in that folder and run ingest again.

You should see an answer and sources with `page` and `document`.

## Phase 1 graph

`retrieve_node` embeds the question and searches Qdrant. `generate_node` builds a grounded prompt and calls Gemini or OpenAI. `run_rag(question)` is a thin `graph.invoke` wrapper.

FastAPI `/ingest` and `/ask` land in phase 2.
