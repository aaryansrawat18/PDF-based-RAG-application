from typing import Any, TypedDict


class RetrievedChunk(TypedDict, total=False):
    text: str
    page: int
    section: str
    document: str
    chunk_id: str
    content_type: str
    score: float


class RAGState(TypedDict, total=False):
    question: str
    rewritten_query: str
    filters: dict[str, Any] | None
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    pruned: list[RetrievedChunk]
    context_ok: bool
    retry_count: int
    generate_retry_count: int
    answer: str
    sources: list[dict[str, Any]]
