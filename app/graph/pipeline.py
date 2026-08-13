from langgraph.graph import END, START, StateGraph

from app.graph.nodes import generate_node, prune_node, rerank_node, retrieve_node
from app.graph.state import RAGState

_graph = None


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("prune", prune_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "prune")
    graph.add_edge("prune", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    """Drop the cached graph. Used in tests after graph shape changes."""
    global _graph
    _graph = None
