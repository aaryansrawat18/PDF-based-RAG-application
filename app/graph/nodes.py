from app.config import settings
from app.core.embeddings import embed_query
from app.core.llm import generate
from app.core.prompts import build_prompt
from app.core.vectorstore import similarity_search
from app.graph.state import RAGState


def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    query_vector = embed_query(question)
    retrieved = similarity_search(query_vector, k=settings.retrieve_k)
    return {"retrieved": retrieved}


def generate_node(state: RAGState) -> dict:
    retrieved = state.get("retrieved") or []
    prompt = build_prompt(state["question"], retrieved)
    answer = generate(prompt)
    sources = [
        {
            "page": chunk.get("page"),
            "document": chunk.get("document"),
            "chunk_id": chunk.get("chunk_id"),
            "content_type": chunk.get("content_type", "text"),
            "score": chunk.get("score"),
        }
        for chunk in retrieved
    ]
    return {"answer": answer, "sources": sources}
