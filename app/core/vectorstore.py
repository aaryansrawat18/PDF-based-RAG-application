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
    PayloadSchemaType,
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


def _collection_vector_size(name: str) -> int | None:
    info = get_client().get_collection(name)
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return int(vectors.size)
    return None


_PAYLOAD_INDEXES = {
    "section": PayloadSchemaType.KEYWORD,
    "document": PayloadSchemaType.KEYWORD,
    "chunk_id": PayloadSchemaType.KEYWORD,
    "content_type": PayloadSchemaType.KEYWORD,
    "page": PayloadSchemaType.INTEGER,
}


def _existing_payload_indexes(name: str) -> set[str]:
    info = get_client().get_collection(name)
    schema = getattr(info, "payload_schema", None) or {}
    return set(schema.keys())


def _ensure_payload_indexes(name: str) -> None:
    client = get_client()
    existing = _existing_payload_indexes(name)
    for field, schema in _PAYLOAD_INDEXES.items():
        if field in existing:
            continue
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            # Index may already exist (local Qdrant / race on re-ingest).
            pass


def ensure_collection(vector_size: int) -> None:
    client = get_client()
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        current = _collection_vector_size(name)
        if current != vector_size:
            # Old BGE (384-d) collections cannot store OpenAI (1536-d) vectors.
            client.delete_collection(name)
            existing.discard(name)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    _ensure_payload_indexes(name)


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
                "section": chunk.get("section") or "Unknown",
                "document": chunk["document"],
                "chunk_id": chunk["chunk_id"],
                "content_type": chunk.get("content_type", "text"),
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


def similarity_search(
    query_vector: list[float],
    k: int | None = None,
    query_filter: Filter | None = None,
) -> list[dict]:
    limit = k if k is not None else settings.retrieve_k
    response = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
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
                "section": payload.get("section"),
                "document": payload.get("document"),
                "chunk_id": payload.get("chunk_id"),
                "content_type": payload.get("content_type", "text"),
                "score": float(point.score),
            }
        )
    return hits
