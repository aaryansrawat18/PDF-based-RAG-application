import unittest

from app.core.pruning import estimate_tokens, prune_chunks, word_set


def _chunk(chunk_id: str, text: str, score: float) -> dict:
    return {
        "document": "rag.pdf",
        "chunk_id": chunk_id,
        "text": text,
        "page": 1,
        "section": "Retrieval",
        "content_type": "text",
        "score": score,
    }


class PruningTests(unittest.TestCase):
    def test_low_score_chunks_are_dropped(self):
        chunks = [
            _chunk("keep", "Hybrid retrieval fuses dense and BM25.", 0.9),
            _chunk("drop", "This passage is off topic.", 0.1),
            _chunk("keep2", "Rerankers score query and chunk pairs.", 0.8),
        ]

        pruned = prune_chunks(chunks, top_k=5, score_threshold=0.5)

        ids = [chunk["chunk_id"] for chunk in pruned]
        self.assertEqual(ids, ["keep", "keep2"])
        self.assertNotIn("drop", ids)

    def test_overlapping_duplicate_is_dropped(self):
        shared = (
            "Retrieval augmented generation fetches passages from a corpus "
            "and then conditions the language model on those passages."
        )
        chunks = [
            _chunk("first", shared, 0.95),
            _chunk("duplicate", shared, 0.90),
            _chunk("unique", "Cross-encoder MiniLM reranks fused candidates.", 0.85),
        ]

        pruned = prune_chunks(
            chunks,
            top_k=5,
            score_threshold=0.0,
            overlap_threshold=0.8,
        )

        ids = [chunk["chunk_id"] for chunk in pruned]
        self.assertEqual(ids[0], "first")
        self.assertNotIn("duplicate", ids)
        self.assertIn("unique", ids)

    def test_token_budget_drops_later_chunks(self):
        chunks = [
            _chunk("short", "Tiny.", 0.9),
            _chunk("long", "word " * 80, 0.8),
        ]

        budget = estimate_tokens("Tiny.") + 5
        pruned = prune_chunks(
            chunks,
            top_k=5,
            score_threshold=0.0,
            max_tokens=budget,
        )

        self.assertEqual([chunk["chunk_id"] for chunk in pruned], ["short"])

    def test_keeps_at_most_top_k(self):
        chunks = [
            _chunk("a", "alpha passage about retrieval.", 0.9),
            _chunk("b", "beta passage about reranking.", 0.8),
            _chunk("c", "gamma passage about pruning.", 0.7),
        ]

        pruned = prune_chunks(chunks, top_k=2, score_threshold=0.0)
        self.assertEqual(len(pruned), 2)
        self.assertEqual([chunk["chunk_id"] for chunk in pruned], ["a", "b"])

    def test_empty_input(self):
        self.assertEqual(prune_chunks([]), [])
        self.assertTrue(word_set("GPT-4o-mini rocks"))


if __name__ == "__main__":
    unittest.main()
