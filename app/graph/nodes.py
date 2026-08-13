from app.config import settings
from app.core.bm25 import search_bm25
from app.core.embeddings import embed_query
from app.core.hybrid import reciprocal_rank_fusion
from app.core.llm import generate
from app.core.metadata import filters_to_qdrant
from app.core.prompts import build_prompt
from app.core.pruning import prune_chunks
from app.core.reranker import rerank
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


def rerank_node(state: RAGState) -> dict:
    """Score fused chunks with a cross-encoder and keep rerank_k."""
    retrieved = state.get("retrieved") or []
    reranked = rerank(state["question"], retrieved)
    return {"reranked": reranked}


def prune_node(state: RAGState) -> dict:
    """Drop low-score, overlapping, and over-budget chunks."""
    reranked = state.get("reranked") or []
    pruned = prune_chunks(reranked)
    return {"pruned": pruned}


def generate_node(state: RAGState) -> dict:
    pruned = state.get("pruned") or []
    prompt = build_prompt(state["question"], pruned)
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
        for chunk in pruned
    ]
    return {"answer": answer, "sources": sources}
