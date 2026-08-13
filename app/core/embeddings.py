from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def _is_bge() -> bool:
    return "bge" in settings.embedding_model.lower()


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 16,
        batch_size=32,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    query = f"{_BGE_QUERY_PREFIX}{text}" if _is_bge() else text
    vector = get_model().encode([query], normalize_embeddings=True)[0]
    return vector.tolist()
