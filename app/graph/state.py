from typing import Any, TypedDict


class RetrievedChunk(TypedDict, total=False):
    text: str
    page: int
    document: str
    chunk_id: str
    content_type: str
    score: float


class RAGState(TypedDict, total=False):
    question: str
    retrieved: list[RetrievedChunk]
    answer: str
    sources: list[dict[str, Any]]
