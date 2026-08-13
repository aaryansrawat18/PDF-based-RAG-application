"""PDF RAG application package.

Two paths matter:

1. INGEST (write) — PDF → pages → chunks → embeddings → Qdrant + BM25
   Entry: POST /ingest  or  python -m app.cli ingest

2. ASK (read) — question → LangGraph → answer + page citations
   Entry: POST /ask  or  python -m app.cli ask "..."

Ask graph (advanced), in order:
  retrieve (vector + BM25 → RRF)
  → rerank (cross-encoder)
  → prune (drop noise / fit token budget)
  → quality_check → rewrite+retrieve again if weak
  → generate → one regenerate if answer is a refusal

Package map:
  app.api       HTTP (FastAPI routes)
  app.graph     LangGraph state, nodes, wiring
  app.core      PDF load, chunk, embed, stores, LLM
  app.eval      Gold-set benchmark + LangSmith tracing
  app.models    Request/response schemas
  app.config    Settings from .env
"""
