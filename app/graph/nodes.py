from app.config import settings
from app.core.bm25 import search_bm25
from app.core.embeddings import embed_query
from app.core.hybrid import reciprocal_rank_fusion
from app.core.llm import generate
from app.core.metadata import filters_to_qdrant
from app.core.prompts import build_prompt
from app.core.vectorstore import similarity_search
from app.graph.state import RAGState


def retrieve_node(state: RAGState) -> dict:
    """Hybrid retrieve: vector search + BM25, then fuse with RRF."""
    question = state["question"]
    filters = state.get("filters")
    top_k = settings.retrieve_k

    query_vector = embed_query(question)
    vector_chunks = similarity_search(
        query_vector,
        k=top_k,
        query_filter=filters_to_qdrant(filters),
    )
    keyword_chunks = search_bm25(question, k=top_k, filters=filters)

    fused_chunks = reciprocal_rank_fusion([vector_chunks, keyword_chunks])
    return {"retrieved": fused_chunks[:top_k]}


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
