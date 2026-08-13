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
    """Pick the query string used for retrieve / rerank.

    Prefers `rewritten_query` after a rewrite loop; otherwise uses the
    original `question`. Empty / whitespace values are treated as missing.
    """
    rewritten = (state.get("rewritten_query") or "").strip()
    if rewritten:
        return rewritten
    return (state.get("question") or "").strip()


def context_is_ok(chunks: list | None) -> bool:
    """Decide whether pruned context is strong enough to generate from.

    Fails when there are too few non-empty chunks, or when the best chunk
    score is below `settings.context_score_threshold`. Used by
    quality_check_node and the rewrite routing edge.
    """
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
    """Decide whether the model answer looks grounded (not a refusal).

    Empty text or matches of `_REFUSAL_RE` count as failures so
    route_after_generate can trigger one regenerate pass.
    """
    text = (answer or "").strip()
    if not text:
        return False
    if _REFUSAL_RE.search(text):
        return False
    return True


def _clean_rewritten_query(text: str, fallback: str) -> str:
    """Normalize LLM rewrite output into a plain search query.

    Strips quotes, keeps the first line, drops labels like "Rewritten query:",
    and falls back to the previous query when the result is too short.
    """
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
    """Hybrid retrieve: dense vector search + BM25, fused with RRF.

    1. Embed `active_query(state)`.
    2. Run Qdrant similarity search and BM25 keyword search (same top_k).
    3. Fuse both ranked lists with reciprocal rank fusion.
    4. Store the top `retrieve_k` hits in `retrieved`.
    """
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
    """Vector-only retrieve for the Phase 1 baseline graph.

    Skips BM25, rerank, and prune. Copies the same hits into `retrieved`,
    `reranked`, and `pruned` so generate_node can run unchanged, and sets
    `context_ok` so the quality/rewrite loop is not needed.
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
    """Rescore fused chunks with a cross-encoder and keep `rerank_k`.

    Takes `retrieved` from hybrid RRF, scores each (query, chunk) pair, and
    writes the reordered top hits to `reranked`.
    """
    retrieved = state.get("retrieved") or []
    reranked = rerank(active_query(state), retrieved)
    return {"reranked": reranked}


def prune_node(state: RAGState) -> dict:
    """Trim reranked chunks into the final LLM context.

    Drops low-score, near-duplicate, and over-budget passages via
    prune_chunks(), then stores the kept list in `pruned`.
    """
    reranked = state.get("reranked") or []
    pruned = prune_chunks(reranked)
    return {"pruned": pruned}


def quality_check_node(state: RAGState) -> dict:
    """Set `context_ok` from the pruned chunk list.

    Downstream routing uses this flag: weak context → rewrite_query;
    strong context (or retries exhausted) → generate.
    """
    return {"context_ok": context_is_ok(state.get("pruned") or [])}


def rewrite_query_node(state: RAGState) -> dict:
    """Rewrite a weak query with the light model, then retrieve again.

    Builds rewrite prompts from the original question, current active query,
    and pruned context. On LLM failure, keeps the previous query. Bumps
    `retry_count` and clears `context_ok` so the graph loops back to retrieve.
    """
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
    """Call the chat model with pruned context and attach citation sources.

    On a regenerate pass (`generate_retry_count` > 0), build_messages uses a
    stricter "use the context" nudge. Returns answer text, source metadata
    for each pruned chunk, and an incremented generate retry counter.
    """
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
