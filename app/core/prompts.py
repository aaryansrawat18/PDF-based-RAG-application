SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Answer using only the provided context. "
    "Context may include prose, markdown tables, and figure captions. "
    "Read tables by matching column headers to row values. "
    "If the context does not contain the answer, say you don't know. "
    "Cite page numbers from the context when you use them."
)


def build_prompt(question: str, retrieved: list[dict]) -> str:
    if not retrieved:
        context = "(no retrieved passages)"
    else:
        parts = []
        for index, chunk in enumerate(retrieved, start=1):
            document = chunk.get("document", "unknown")
            page = chunk.get("page", "?")
            section = chunk.get("section") or "Unknown"
            content_type = chunk.get("content_type", "text")
            text = chunk.get("text", "")
            parts.append(
                f"[{index}] document={document} page={page} "
                f"section={section} type={content_type}\n{text}"
            )
        context = "\n\n".join(parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
