import time

from app.core.pruning import estimate_tokens
from app.eval.tracing import traced_invoke
from app.graph.pipeline import get_baseline_graph, get_graph

_PIPELINES = {"advanced", "baseline"}


def _pages(chunks: list | None) -> list[int]:
    pages: list[int] = []
    for chunk in chunks or []:
        page = chunk.get("page")
        if page is None:
            continue
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return pages


def _pack_result(result: dict, latency_ms: int, pipeline: str) -> dict:
    retrieved = result.get("retrieved") or []
    pruned = result.get("pruned") or retrieved
    sources = result.get("sources") or []
    context_chunks = pruned or retrieved
    context_text = " ".join(chunk.get("text") or "" for chunk in context_chunks)
    answer = result.get("answer") or ""
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "rewritten_query": result.get("rewritten_query") or None,
        "retry_count": int(result.get("retry_count") or 0),
        "pipeline": pipeline,
        "retrieved": retrieved,
        "pruned": pruned,
        "retrieved_pages": _pages(retrieved),
        "source_pages": _pages(sources) or _pages(pruned),
        "context_tokens": estimate_tokens(context_text) if context_text else 0,
        "answer_tokens": estimate_tokens(answer) if answer else 0,
    }


def run_rag(
    question: str,
    filters: dict | None = None,
    *,
    pipeline: str = "advanced",
) -> dict:
    """Thin wrapper around graph.invoke.

    pipeline=advanced: hybrid → rerank → prune → quality → generate.
    pipeline=baseline: vector retrieve → generate (Phase 1, for eval).
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    name = (pipeline or "advanced").strip().lower()
    if name not in _PIPELINES:
        raise ValueError(f"pipeline must be one of {sorted(_PIPELINES)}, got {pipeline!r}")

    payload: dict = {
        "question": question.strip(),
        "retry_count": 0,
        "generate_retry_count": 0,
        "context_ok": False,
        "rewritten_query": "",
    }
    if filters:
        payload["filters"] = filters

    graph = get_baseline_graph() if name == "baseline" else get_graph()
    started = time.perf_counter()
    result = traced_invoke(graph, payload, pipeline=name)
    latency_ms = int((time.perf_counter() - started) * 1000)
    return _pack_result(result, latency_ms, name)
