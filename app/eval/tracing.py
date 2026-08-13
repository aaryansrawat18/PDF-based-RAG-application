"""LangSmith env + wrap graph.invoke / the OpenAI client.

Prompt cache is the OpenAI prefix KV-cache (see app.core.llm). This module
only turns on traces so LangSmith shows retrieve, rerank, prune, generate,
and rewrite/retry spans plus token / cached_token usage.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import settings

_CONFIGURED = False


def tracing_enabled() -> bool:
    """True when a LangSmith key is present and tracing is not turned off."""
    if not (settings.langsmith_api_key or os.getenv("LANGSMITH_API_KEY")):
        return False
    return bool(settings.langsmith_tracing)


def configure_tracing() -> None:
    """Copy settings into process env so LangGraph and wrap_openai auto-trace.

    Sets both LANGSMITH_* and legacy LANGCHAIN_TRACING_V2 aliases.
    Safe to call more than once.
    """
    global _CONFIGURED

    api_key = settings.langsmith_api_key or os.getenv("LANGSMITH_API_KEY", "")
    project = settings.langsmith_project or os.getenv("LANGSMITH_PROJECT", "rag-pipeline")
    endpoint = settings.langsmith_endpoint or os.getenv("LANGSMITH_ENDPOINT", "")
    enabled = tracing_enabled()
    flag = "true" if enabled else "false"

    os.environ["LANGSMITH_TRACING"] = flag
    os.environ["LANGCHAIN_TRACING_V2"] = flag
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_PROJECT"] = project
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint.rstrip("/")
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint.rstrip("/")

    _CONFIGURED = True


def wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI client so chat.completions spans include token usage."""
    configure_tracing()
    try:
        from langsmith.wrappers import wrap_openai
    except ImportError:
        return client
    return wrap_openai(client)


def graph_run_config(payload: dict, *, pipeline: str = "advanced") -> dict:
    """LangGraph invoke config: run name, tags, and light metadata."""
    if pipeline == "baseline":
        tags = ["rag", "baseline", "eval"]
        run_name = "rag_ask_baseline"
    else:
        tags = ["rag", "hybrid", "rerank", "prune"]
        run_name = "rag_ask"
    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": {
            "has_filters": bool(payload.get("filters")),
            "prompt_cache_key": settings.prompt_cache_key,
            "pipeline": pipeline,
        },
    }


def traced_invoke(
    graph: Any,
    payload: dict,
    *,
    pipeline: str = "advanced",
) -> dict:
    """Invoke the compiled graph. Parent span is rag_ask when tracing is on."""
    configure_tracing()
    config = graph_run_config(payload, pipeline=pipeline)
    run_name = config["run_name"]

    def _invoke(state: dict) -> dict:
        return graph.invoke(state, config=config)

    if not tracing_enabled():
        return _invoke(payload)

    try:
        from langsmith import traceable
    except ImportError:
        return _invoke(payload)

    wrapped = traceable(name=run_name, run_type="chain")(_invoke)
    return wrapped(payload)


def flush_traces() -> None:
    """Block until queued LangSmith runs are sent. Use on CLI shutdown."""
    if not tracing_enabled():
        return
    waiters = []
    for import_path in (
        ("langsmith", "wait_for_all_tracers"),
        ("langsmith.run_trees", "wait_for_all_tracers"),
        ("langchain_core.tracers.langchain", "wait_for_all_tracers"),
    ):
        module_name, attr = import_path
        try:
            module = __import__(module_name, fromlist=[attr])
            waiters.append(getattr(module, attr))
            break
        except (ImportError, AttributeError):
            continue
    if waiters:
        waiters[0]()
