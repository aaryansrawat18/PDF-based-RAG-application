"""BM25 keyword search.

We keep a JSON file of all ingested chunks (the corpus).
On each search we load that file, score chunks with BM25, and return the top hits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.metadata import chunk_matches_filters

# Fields we store for each chunk. Enough to search and to show sources later.
_CORPUS_FIELDS = ("text", "page", "section", "document", "chunk_id", "content_type")

# "GPT-4o-mini" stays one word. Spaces and punctuation split the rest.
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize_text(text: str) -> list[str]:
    """Turn text into lowercase words for BM25."""
    return _WORD_PATTERN.findall((text or "").lower())


def bm25_corpus_path() -> Path:
    return Path(settings.bm25_corpus_path)


def load_bm25_corpus() -> list[dict]:
    path = bm25_corpus_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        return []
    return data


def save_bm25_corpus(chunks: list[dict]) -> None:
    path = bm25_corpus_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def _chunk_for_corpus(chunk: dict) -> dict:
    return {field: chunk.get(field) for field in _CORPUS_FIELDS}


def update_bm25_corpus(document: str, new_chunks: list[dict]) -> None:
    """Replace one document's chunks in the saved corpus, then write the file.

    Called during ingest. Other documents are left as they are.
    """
    kept_chunks = [
        chunk for chunk in load_bm25_corpus() if chunk.get("document") != document
    ]
    kept_chunks.extend(_chunk_for_corpus(chunk) for chunk in new_chunks)
    save_bm25_corpus(kept_chunks)


def search_bm25(
    query: str,
    k: int | None = None,
    filters: dict | None = None,
    corpus: list[dict] | None = None,
) -> list[dict]:
    """Return the top-k chunks that match the query by keyword.

    Pass `corpus` in tests. Live /ask loads the saved JSON file.
    """
    limit = k if k is not None else settings.retrieve_k
    chunks = corpus if corpus is not None else load_bm25_corpus()
    query_words = tokenize_text(query)
    if not chunks or not query_words or limit <= 0:
        return []

    chunk_words = [tokenize_text(chunk.get("text") or "") for chunk in chunks]
    if not any(chunk_words):
        return []

    bm25 = BM25Okapi(chunk_words)
    scores = bm25.get_scores(query_words)
    query_word_set = set(query_words)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    hits: list[dict] = []
    for index in ranked_indexes:
        # Skip chunks that share no words with the query.
        if query_word_set.isdisjoint(chunk_words[index]):
            continue
        chunk = chunks[index]
        if filters and not chunk_matches_filters(chunk, filters):
            continue
        hit = dict(chunk)
        hit["score"] = float(scores[index])
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits
