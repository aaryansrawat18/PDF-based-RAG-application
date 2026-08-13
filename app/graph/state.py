"""Shared LangGraph state for one ask request.

Each node reads fields it needs and returns a partial dict that merges
into this state. Flow of list fields:

  retrieved  (hybrid RRF, ~retrieve_k)
       ↓
  reranked   (cross-encoder, ~rerank_k)
       ↓
  pruned     (final LLM context, ~prune_k)
       ↓
  sources    (citation metadata copied at generate time)
"""

from typing import Any, TypedDict


class RetrievedChunk(TypedDict, total=False):
    """One passage from Qdrant and/or BM25 (same shape after fusion)."""

    text: str
    page: int
    section: str
    document: str
    chunk_id: str
    content_type: str
    score: float


class RAGState(TypedDict, total=False):
    """Keys set across the ask graph. total=False → nodes may omit fields."""

    # Inputs
    question: str
    rewritten_query: str
    filters: dict[str, Any] | None

    # Retrieval funnel
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    pruned: list[RetrievedChunk]

    # Control flags for conditional edges
    context_ok: bool
    retry_count: int  # rewrite → retrieve loops
    generate_retry_count: int  # refusal regenerate loops

    # Outputs
    answer: str
    sources: list[dict[str, Any]]
