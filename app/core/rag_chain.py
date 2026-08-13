from app.graph.pipeline import get_graph


def run_rag(question: str, filters: dict | None = None) -> dict:
    """Thin wrapper: invoke the LangGraph hybrid-retrieve → generate pipeline."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    payload: dict = {"question": question.strip()}
    if filters:
        payload["filters"] = filters

    result = get_graph().invoke(payload)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }
