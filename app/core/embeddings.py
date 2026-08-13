from functools import lru_cache

from openai import OpenAI

from app.config import settings

# OpenAI allows up to 2048 inputs per embeddings request.
_EMBED_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and set it.")
    return OpenAI(api_key=settings.openai_api_key)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = _client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + _EMBED_BATCH_SIZE]))
    return vectors


def embed_query(text: str) -> list[float]:
    return _embed_batch([text])[0]
