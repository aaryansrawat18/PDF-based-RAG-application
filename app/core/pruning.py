"""Context pruning.

After rerank, drop chunks the LLM should not see: low scores,
near-duplicate overlap, and anything past the token budget.
"""

from __future__ import annotations

import re

from app.config import settings

_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def word_set(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall((text or "").lower()))


def jaccard_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def prune_chunks(
    chunks: list[dict],
    top_k: int | None = None,
    score_threshold: float | None = None,
    overlap_threshold: float | None = None,
    max_tokens: int | None = None,
) -> list[dict]:
    """Keep the best unique chunks that fit the token budget.

    Chunks are assumed already sorted by score (rerank output).
    """
    limit = settings.prune_k if top_k is None else top_k
    min_score = (
        settings.prune_score_threshold
        if score_threshold is None
        else score_threshold
    )
    max_overlap = (
        settings.prune_overlap_threshold
        if overlap_threshold is None
        else overlap_threshold
    )
    token_budget = (
        settings.prune_max_tokens if max_tokens is None else max_tokens
    )

    if not chunks or limit <= 0:
        return []

    kept: list[dict] = []
    kept_word_sets: list[set[str]] = []
    used_tokens = 0

    for chunk in chunks:
        score = chunk.get("score")
        if score is not None and float(score) < min_score:
            continue

        text = chunk.get("text") or ""
        words = word_set(text)
        if any(
            jaccard_overlap(words, existing) >= max_overlap
            for existing in kept_word_sets
        ):
            continue

        tokens = estimate_tokens(text)
        if kept and used_tokens + tokens > token_budget:
            continue

        kept.append(dict(chunk))
        kept_word_sets.append(words)
        used_tokens += tokens
        if len(kept) >= limit:
            break

    return kept
