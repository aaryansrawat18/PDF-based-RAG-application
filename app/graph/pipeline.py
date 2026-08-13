"""Wire LangGraph nodes into the advanced and baseline RAG graphs.

Advanced flow (POST /ask default):

  START
    → retrieve      hybrid: dense (Qdrant) + sparse (BM25) → RRF
    → rerank        cross-encoder scores (query, chunk)
    → prune         drop low-score / duplicate / over-budget chunks
    → quality_check set context_ok from pruned scores
         │
         ├─ context weak & retries left → rewrite_query → retrieve (loop)
         └─ else → generate
                      │
                      ├─ refusal + has context → generate again (once)
                      └─ else → END

Baseline flow (eval only): START → vector retrieve → generate → END
"""

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.nodes import (
    answer_is_ok,
    baseline_retrieve_node,
    generate_node,
    prune_node,
    quality_check_node,
    rerank_node,
    retrieve_node,
    rewrite_query_node,
)
from app.graph.state import RAGState

_graph = None
_baseline_graph = None


def route_after_quality(state: RAGState) -> str:
    """Poor context → rewrite (until retry cap); otherwise generate."""
    if state.get("context_ok"):
        return "generate"
    if int(state.get("retry_count") or 0) >= settings.max_retrieve_retries:
        return "generate"
    return "rewrite"


def route_after_generate(state: RAGState) -> str:
    """Poor answer with usable context → regenerate once; otherwise end."""
    attempts = int(state.get("generate_retry_count") or 0)
    has_context = bool(state.get("pruned"))
    if (
        not answer_is_ok(state.get("answer") or "")
        and has_context
        and attempts <= settings.max_generate_retries
    ):
        return "retry"
    return "end"


def build_graph():
    """Compile the full hybrid → rerank → prune → rewrite/retry graph."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("prune", prune_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)

    # Linear funnel first…
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "prune")
    graph.add_edge("prune", "quality_check")

    # …then optional rewrite loop back to retrieve.
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve")

    # Optional one-shot regenerate on refusal.
    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"retry": "generate", "end": END},
    )
    return graph.compile()


def get_graph():
    """Cached advanced graph (one compile per process)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def build_baseline_graph():
    """Phase 1 linear graph: vector retrieve → generate. No hybrid/rerank/prune."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", baseline_retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def get_baseline_graph():
    """Cached baseline graph used by eval comparisons."""
    global _baseline_graph
    if _baseline_graph is None:
        _baseline_graph = build_baseline_graph()
    return _baseline_graph


def reset_graph():
    """Drop cached graphs. Used in tests after graph shape changes."""
    global _graph, _baseline_graph
    _graph = None
    _baseline_graph = None
