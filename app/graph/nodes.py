from app.config import settings
from app.core.embeddings import embed_query
from app.core.llm import generate
from app.core.metadata import filters_to_qdrant
from app.core.prompts import build_prompt
from app.core.vectorstore import similarity_search
from app.graph.state import RAGState


def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    query_vector = embed_query(question)
    query_filter = filters_to_qdrant(state.get("filters"))
    retrieved = similarity_search(
        query_vector,
        k=settings.retrieve_k,
        query_filter=query_filter,
    )
    return {"retrieved": retrieved}


def generate_node(state: RAGState) -> dict:
    retrieved = state.get("retrieved") or []
    prompt = build_prompt(state["question"], retrieved)
    answer = generate(prompt)
    sources = [
        {
            "document": chunk.get("document"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_id": chunk.get("chunk_id"),
            "score": chunk.get("score"),
            "content_type": chunk.get("content_type", "text"),
        }
        for chunk in retrieved
    ]
    return {"answer": answer, "sources": sources}
