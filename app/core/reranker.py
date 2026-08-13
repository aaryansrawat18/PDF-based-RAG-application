"""Cross-encoder reranker.

Hybrid RRF is a cheap first pass. A cross-encoder then scores each
(query, chunk) pair together and reorders the fused list so the LLM
sees the most relevant passages first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from app.config import settings

_cross_encoder = None


def get_cross_encoder():
    """Load the MiniLM cross-encoder once per process."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(settings.reranker_model)
    return _cross_encoder


def _as_float_list(scores) -> list[float]:
    if scores is None:
        return []
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(score) for score in scores]


def cross_encoder_scores(query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs. Higher is more relevant."""
    if not texts:
        return []
    pairs = [(query, text or "") for text in texts]
    raw_scores = get_cross_encoder().predict(pairs)
    return _as_float_list(raw_scores)


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
    score_fn: Callable[[str, list[str]], Sequence[float]] | None = None,
) -> list[dict]:
    """Reorder chunks by cross-encoder score and keep the top `top_k`.

    Pass `score_fn` in tests so we do not download the model.
    """
    limit = settings.rerank_k if top_k is None else top_k
    if not chunks or limit <= 0:
        return []

    texts = [chunk.get("text") or "" for chunk in chunks]
    scorer = score_fn or cross_encoder_scores
    raw_scores = list(scorer(query, texts))
    if len(raw_scores) != len(chunks):
        raise ValueError(
            f"score_fn returned {len(raw_scores)} scores for {len(chunks)} chunks"
        )

    ranked: list[dict] = []
    for chunk, score in zip(chunks, raw_scores, strict=True):
        updated = dict(chunk)
        updated["score"] = float(score)
        ranked.append(updated)

    ranked.sort(key=lambda chunk: chunk["score"], reverse=True)
    return ranked[:limit]
