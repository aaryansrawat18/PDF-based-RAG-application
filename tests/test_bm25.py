import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import settings
from app.core.bm25 import (
    load_bm25_corpus,
    search_bm25,
    tokenize_text,
    update_bm25_corpus,
)


class TokenizeTests(unittest.TestCase):
    def test_keeps_hyphenated_model_names(self):
        words = tokenize_text("The model GPT-4o-mini is small.")
        self.assertIn("gpt-4o-mini", words)
        self.assertIn("model", words)

    def test_empty_text(self):
        self.assertEqual(tokenize_text(""), [])


class Bm25SearchTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {
                "text": "Use GPT-4o-mini for cheap generation.",
                "page": 2,
                "section": "Models",
                "document": "rag.pdf",
                "chunk_id": "chunk_0",
                "content_type": "text",
            },
            {
                "text": "Hybrid search fuses dense vectors with BM25.",
                "page": 12,
                "section": "Retrieval",
                "document": "rag.pdf",
                "chunk_id": "chunk_1",
                "content_type": "text",
            },
            {
                "text": "Tables store labeled rows for lookup.",
                "page": 8,
                "section": "Chunking",
                "document": "other.pdf",
                "chunk_id": "chunk_0",
                "content_type": "table",
            },
        ]

    def test_exact_term_ranks_first(self):
        hits = search_bm25("GPT-4o-mini", k=3, corpus=self.corpus)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["chunk_id"], "chunk_0")
        self.assertIn("GPT-4o-mini", hits[0]["text"])

    def test_keyword_finds_bm25_chunk(self):
        hits = search_bm25("BM25 keyword", k=3, corpus=self.corpus)
        self.assertEqual(hits[0]["chunk_id"], "chunk_1")

    def test_filters_limit_results(self):
        hits = search_bm25(
            "BM25",
            k=5,
            filters={"section": "Retrieval"},
            corpus=self.corpus,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["section"], "Retrieval")
        self.assertEqual(hits[0]["chunk_id"], "chunk_1")

    def test_empty_query_returns_empty(self):
        self.assertEqual(search_bm25("???", k=3, corpus=self.corpus), [])


class Bm25CorpusTests(unittest.TestCase):
    def test_update_replaces_one_document_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_file = Path(tmp) / "bm25_corpus.json"
            with patch.object(settings, "bm25_corpus_path", str(corpus_file)):
                update_bm25_corpus(
                    "a.pdf",
                    [
                        {
                            "text": "old a",
                            "page": 1,
                            "section": "Intro",
                            "document": "a.pdf",
                            "chunk_id": "chunk_0",
                            "content_type": "text",
                        }
                    ],
                )
                update_bm25_corpus(
                    "b.pdf",
                    [
                        {
                            "text": "keep b",
                            "page": 1,
                            "section": "Intro",
                            "document": "b.pdf",
                            "chunk_id": "chunk_0",
                            "content_type": "text",
                        }
                    ],
                )
                update_bm25_corpus(
                    "a.pdf",
                    [
                        {
                            "text": "new a",
                            "page": 2,
                            "section": "Intro",
                            "document": "a.pdf",
                            "chunk_id": "chunk_0",
                            "content_type": "text",
                        }
                    ],
                )

                corpus = load_bm25_corpus()
                texts = {chunk["document"]: chunk["text"] for chunk in corpus}
                self.assertEqual(texts["a.pdf"], "new a")
                self.assertEqual(texts["b.pdf"], "keep b")
                self.assertTrue(corpus_file.exists())
                saved = json.loads(corpus_file.read_text(encoding="utf-8"))
                self.assertEqual(len(saved), 2)


if __name__ == "__main__":
    unittest.main()
