"""Reciprocal Rank Fusion (RRF).

Vector search and BM25 each return their own ranked list.
RRF merges those lists into one. A chunk that appears in both lists
gets a higher score than a chunk that appears in only one list.

score = 1/(60 + rank) + 1/(60 + rank) + ...
Rank is 1 for the first hit in a list, 2 for the second, and so on.
"""

from __future__ import annotations

from app.config import settings


def make_chunk_key(chunk: dict) -> str:
    """Unique id for a chunk across documents."""
    document = chunk.get("document") or ""
    chunk_id = chunk.get("chunk_id") or ""
    return f"{document}::{chunk_id}"


def rrf_score_for_rank(rank: int, rrf_k: int | None = None) -> float:
    """Convert a 1-based rank into an RRF score."""
    k = settings.rrf_k if rrf_k is None else rrf_k
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    rrf_k: int | None = None,
) -> list[dict]:
    """Merge ranked lists. Each chunk appears once, sorted by fused score."""
    k = settings.rrf_k if rrf_k is None else rrf_k
    fused_scores: dict[str, float] = {}
    chunk_by_key: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = make_chunk_key(chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score_for_rank(rank, k)
            if key not in chunk_by_key:
                chunk_by_key[key] = chunk

    sorted_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    fused: list[dict] = []
    for key in sorted_keys:
        chunk = dict(chunk_by_key[key])
        chunk["score"] = fused_scores[key]
        fused.append(chunk)
    return fused
