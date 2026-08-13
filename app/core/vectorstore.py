from __future__ import annotations

import atexit
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return QdrantClient(path=settings.qdrant_path)


def ensure_collection(vector_size: int) -> None:
    client = get_client()
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _point_id(document: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document}:{chunk_id}"))


def delete_document(document: str) -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        return
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document", match=MatchValue(value=document))]
        ),
    )


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")
    if not chunks:
        return 0

    ensure_collection(len(embeddings[0]))
    documents = {c["document"] for c in chunks}
    for document in documents:
        delete_document(document)

    points = [
        PointStruct(
            id=_point_id(chunk["document"], chunk["chunk_id"]),
            vector=vector,
            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "document": chunk["document"],
                "chunk_id": chunk["chunk_id"],
            },
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]
    get_client().upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def close_client() -> None:
    if get_client.cache_info().currsize == 0:
        return
    try:
        get_client().close()
    except Exception:
        pass
    finally:
        get_client.cache_clear()


atexit.register(close_client)


def similarity_search(query_vector: list[float], k: int | None = None) -> list[dict]:
    limit = k if k is not None else settings.retrieve_k
    response = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    hits: list[dict] = []
    for point in response.points:
        payload = point.payload or {}
        hits.append(
            {
                "text": payload.get("text", ""),
                "page": payload.get("page"),
                "document": payload.get("document"),
                "chunk_id": payload.get("chunk_id"),
                "score": float(point.score),
            }
        )
    return hits
