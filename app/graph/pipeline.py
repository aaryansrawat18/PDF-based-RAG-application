from langgraph.graph import END, START, StateGraph

from app.graph.nodes import generate_node, retrieve_node
from app.graph.state import RAGState

_graph = None


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
