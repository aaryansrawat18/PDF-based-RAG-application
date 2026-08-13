"""Retrieval, generation, and system metrics for baseline vs advanced RAG."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from app.core.pruning import estimate_tokens

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
)

RETRIEVAL_METRICS = ("hit_rate", "recall", "precision", "mrr")
GENERATION_METRICS = ("faithfulness", "relevance")
SYSTEM_METRICS = ("latency_ms", "context_tokens", "answer_tokens")
CONTEXT_METRICS = ("context_hit_rate", "context_recall")
SUMMARY_METRICS = (
    *RETRIEVAL_METRICS,
    *CONTEXT_METRICS,
    *GENERATION_METRICS,
    *SYSTEM_METRICS,
)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1]


def as_pages(pages: Iterable[Any] | None) -> list[int]:
    out: list[int] = []
    for page in pages or []:
        try:
            out.append(int(page))
        except (TypeError, ValueError):
            continue
    return out


def unique_pages(pages: Iterable[Any] | None) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for page in as_pages(pages):
        if page in seen:
            continue
        seen.add(page)
        ordered.append(page)
    return ordered


def hit_rate(predicted: Iterable[Any] | None, expected: Iterable[Any] | None) -> float:
    """1.0 if any gold page appears in the predicted list."""
    gold = set(as_pages(expected))
    if not gold:
        return 0.0
    return 1.0 if gold & set(as_pages(predicted)) else 0.0


def recall(predicted: Iterable[Any] | None, expected: Iterable[Any] | None) -> float:
    gold = set(as_pages(expected))
    if not gold:
        return 0.0
    return len(gold & set(as_pages(predicted))) / len(gold)


def precision(predicted: Iterable[Any] | None, expected: Iterable[Any] | None) -> float:
    predicted_pages = set(as_pages(predicted))
    if not predicted_pages:
        return 0.0
    gold = set(as_pages(expected))
    return len(gold & predicted_pages) / len(predicted_pages)


def mrr(ranked: Iterable[Any] | None, expected: Iterable[Any] | None) -> float:
    """Mean Reciprocal Rank of the first gold page in a ranked list."""
    gold = set(as_pages(expected))
    if not gold:
        return 0.0
    for rank, page in enumerate(as_pages(ranked), start=1):
        if page in gold:
            return 1.0 / rank
    return 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style bag-of-tokens F1 between prediction and reference."""
    pred = Counter(content_tokens(prediction))
    ref = Counter(content_tokens(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision_score = overlap / sum(pred.values())
    recall_score = overlap / sum(ref.values())
    return 2 * precision_score * recall_score / (precision_score + recall_score)


def faithfulness(answer: str, contexts: Iterable[str] | None) -> float:
    """Share of answer content tokens that also appear in retrieved context."""
    context_vocab = set(content_tokens(" ".join(contexts or [])))
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    if not context_vocab:
        return 0.0
    return sum(1 for token in answer_tokens if token in context_vocab) / len(answer_tokens)


def relevance(answer: str, reference_answer: str) -> float:
    """Lexical overlap with the gold answer (token F1)."""
    return token_f1(answer, reference_answer)


def _context_texts(result: dict) -> list[str]:
    chunks = result.get("pruned") or result.get("retrieved") or []
    texts = [chunk.get("text") or "" for chunk in chunks if isinstance(chunk, dict)]
    if texts:
        return texts
    return [source.get("text") or "" for source in result.get("sources") or [] if isinstance(source, dict)]


def _result_pages(result: dict, *keys: str) -> list[int]:
    for key in keys:
        if key in result and result[key] is not None:
            return as_pages(result[key])
    return []


def score_prediction(example: dict, result: dict) -> dict:
    """Score one pipeline run against a gold eval example."""
    expected = example.get("expected_pages") or []
    retrieved_pages = _result_pages(result, "retrieved_pages")
    if not retrieved_pages:
        retrieved_pages = as_pages(
            chunk.get("page") for chunk in (result.get("retrieved") or []) if isinstance(chunk, dict)
        )
    source_pages = _result_pages(result, "source_pages")
    if not source_pages:
        source_pages = as_pages(
            item.get("page") for item in (result.get("sources") or []) if isinstance(item, dict)
        )
    if not source_pages:
        source_pages = as_pages(
            chunk.get("page") for chunk in (result.get("pruned") or []) if isinstance(chunk, dict)
        )

    answer = result.get("answer") or ""
    contexts = _context_texts(result)
    context_tokens = result.get("context_tokens")
    if context_tokens is None:
        context_tokens = estimate_tokens(" ".join(contexts)) if contexts else 0
    answer_tokens = result.get("answer_tokens")
    if answer_tokens is None:
        answer_tokens = estimate_tokens(answer) if answer else 0

    return {
        "id": example.get("id"),
        "question": example.get("question"),
        "pipeline": result.get("pipeline"),
        "hit_rate": hit_rate(retrieved_pages, expected),
        "recall": recall(retrieved_pages, expected),
        "precision": precision(retrieved_pages, expected),
        "mrr": mrr(retrieved_pages, expected),
        "context_hit_rate": hit_rate(source_pages, expected),
        "context_recall": recall(source_pages, expected),
        "faithfulness": faithfulness(answer, contexts),
        "relevance": relevance(answer, example.get("reference_answer") or ""),
        "latency_ms": int(result.get("latency_ms") or 0),
        "context_tokens": int(context_tokens or 0),
        "answer_tokens": int(answer_tokens or 0),
        "answer": answer,
        "retrieved_pages": retrieved_pages,
        "source_pages": source_pages,
        "expected_pages": as_pages(expected),
        "rewritten_query": result.get("rewritten_query"),
        "retry_count": int(result.get("retry_count") or 0),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_scores(rows: list[dict]) -> dict[str, float]:
    """Mean of each metric over scored rows."""
    summary: dict[str, float] = {"n": float(len(rows))}
    for metric in SUMMARY_METRICS:
        summary[metric] = round(_mean([float(row.get(metric) or 0) for row in rows]), 4)
    return summary


def comparison_table(
    summaries: dict[str, dict[str, float]],
    *,
    baseline: str = "baseline",
    advanced: str = "advanced",
) -> list[dict]:
    """One row per metric: baseline, advanced, and signed delta (advanced - baseline)."""
    left = summaries.get(baseline) or {}
    right = summaries.get(advanced) or {}
    rows = []
    for metric in SUMMARY_METRICS:
        base_value = float(left.get(metric) or 0)
        adv_value = float(right.get(metric) or 0)
        rows.append(
            {
                "metric": metric,
                "baseline": round(base_value, 4),
                "advanced": round(adv_value, 4),
                "delta": round(adv_value - base_value, 4),
            }
        )
    return rows


def format_comparison_table(rows: list[dict]) -> str:
    """Plain-text table for CLI / notebook printouts."""
    header = f"{'metric':<20} {'baseline':>12} {'advanced':>12} {'delta':>12}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['metric']:<20} {row['baseline']:>12.4f} {row['advanced']:>12.4f} {row['delta']:>+12.4f}"
        )
    return "\n".join(lines)
