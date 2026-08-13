"""LangGraph node functions for the ask path.

Each public *_node(state) → dict is a graph step. Helpers (active_query,
context_is_ok, answer_is_ok) support routing and rewrite.

Order in the advanced graph: retrieve → rerank → prune → quality_check
→ (optional rewrite → retrieve) → generate → (optional generate again).
"""

import re

from app.config import settings
from app.core.bm25 import search_bm25
from app.core.embeddings import embed_query
from app.core.hybrid import reciprocal_rank_fusion
from app.core.llm import generate
from app.core.metadata import filters_to_qdrant
from app.core.prompts import build_messages, build_rewrite_messages
from app.core.pruning import prune_chunks
from app.core.reranker import rerank
from app.core.vectorstore import similarity_search
from app.graph.state import RAGState

# Rough refusal detector for route_after_generate (not shown to the user).
_REFUSAL_RE = re.compile(
    r"^\s*(i\s+(do\s+not|don't|cannot|can't)\s+know|"
    r"i\s+(cannot|can't|am unable to)\s+(answer|find|determine)|"
    r"the (provided )?context (does not|doesn't) (contain|include|have)|"
    r"not enough (context|information)|"
    r"no (relevant )?(context|information|passages))",
    re.IGNORECASE,
)


def active_query(state: RAGState) -> str:
    """Search query: rewritten if present, otherwise the original question."""
    rewritten = (state.get("rewritten_query") or "").strip()
    if rewritten:
        return rewritten
    return (state.get("question") or "").strip()


def context_is_ok(chunks: list | None) -> bool:
    """True when pruned context is non-empty and the best score clears the bar."""
    if not chunks:
        return False
    nonempty = [chunk for chunk in chunks if (chunk.get("text") or "").strip()]
    if len(nonempty) < settings.context_min_chunks:
        return False
    scores = [
        float(chunk["score"])
        for chunk in nonempty
        if chunk.get("score") is not None
    ]
    if scores and max(scores) < settings.context_score_threshold:
        return False
    return True


def answer_is_ok(answer: str) -> bool:
    """True when the model produced a grounded answer instead of a refusal."""
    text = (answer or "").strip()
    if not text:
        return False
    if _REFUSAL_RE.search(text):
        return False
    return True


def _clean_rewritten_query(text: str, fallback: str) -> str:
    cleaned = (text or "").strip().strip('"').strip("'")
    if cleaned:
        cleaned = cleaned.splitlines()[0].strip().strip('"').strip("'")
    lowered = cleaned.lower()
    for prefix in ("rewritten query:", "search query:", "query:", "rewrite:"):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip().strip('"').strip("'")
            break
    if len(cleaned) < 3:
        return fallback
    return cleaned


def retrieve_node(state: RAGState) -> dict:
    """Hybrid retrieve: vector search + BM25, then fuse with RRF."""
    question = active_query(state)
    filters = state.get("filters")
    top_k = settings.retrieve_k

    # Dense (semantic) + sparse (keyword) in parallel lists, then fuse.
    query_vector = embed_query(question)
    vector_chunks = similarity_search(
        query_vector,
        k=top_k,
        query_filter=filters_to_qdrant(filters),
    )
    keyword_chunks = search_bm25(question, k=top_k, filters=filters)

    fused_chunks = reciprocal_rank_fusion([vector_chunks, keyword_chunks])
    return {"retrieved": fused_chunks[:top_k]}


def baseline_retrieve_node(state: RAGState) -> dict:
    """Vector-only retrieve (Phase 1 baseline). No BM25, rerank, or prune.

    Hits are copied into pruned so generate_node can run unchanged.
    """
    question = (state.get("question") or "").strip()
    filters = state.get("filters")
    top_k = settings.retrieve_k
    query_vector = embed_query(question)
    chunks = similarity_search(
        query_vector,
        k=top_k,
        query_filter=filters_to_qdrant(filters),
    )
    return {
        "retrieved": chunks,
        "reranked": chunks,
        "pruned": chunks,
        "context_ok": True,
    }


def rerank_node(state: RAGState) -> dict:
    """Score fused chunks with a cross-encoder and keep rerank_k."""
    retrieved = state.get("retrieved") or []
    reranked = rerank(active_query(state), retrieved)
    return {"reranked": reranked}


def prune_node(state: RAGState) -> dict:
    """Drop low-score, overlapping, and over-budget chunks."""
    reranked = state.get("reranked") or []
    pruned = prune_chunks(reranked)
    return {"pruned": pruned}


def quality_check_node(state: RAGState) -> dict:
    """Mark whether pruned context is strong enough to generate from."""
    return {"context_ok": context_is_ok(state.get("pruned") or [])}


def rewrite_query_node(state: RAGState) -> dict:
    """Rewrite a weak query with the light model, then retrieve again."""
    question = (state.get("question") or "").strip()
    previous = active_query(state) or question
    pruned = state.get("pruned") or []
    retry_count = int(state.get("retry_count") or 0) + 1
    messages = build_rewrite_messages(question, previous, pruned)
    try:
        raw = generate(messages=messages, model=settings.rewrite_model)
        rewritten = _clean_rewritten_query(raw, previous)
    except Exception:
        rewritten = previous
    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count,
        "context_ok": False,
    }


def generate_node(state: RAGState) -> dict:
    """Call the chat model with pruned context; attach citation sources."""
    pruned = state.get("pruned") or []
    attempts = int(state.get("generate_retry_count") or 0)
    messages = build_messages(
        state["question"],
        pruned,
        retry=attempts > 0,  # stricter "use the context" nudge on regenerate
    )
    answer = generate(messages=messages)
    sources = [
        {
            "document": chunk.get("document"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_id": chunk.get("chunk_id"),
            "score": chunk.get("score"),
            "content_type": chunk.get("content_type", "text"),
        }
        for chunk in pruned
    ]
    return {
        "answer": answer,
        "sources": sources,
        "generate_retry_count": attempts + 1,
    }
