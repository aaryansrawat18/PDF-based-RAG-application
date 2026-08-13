from app.graph.pipeline import get_graph


def run_rag(question: str) -> dict:
    """Thin wrapper: invoke the LangGraph retrieve → generate pipeline."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    result = get_graph().invoke({"question": question.strip()})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
