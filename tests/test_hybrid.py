import unittest

from app.core.hybrid import make_chunk_key, reciprocal_rank_fusion, rrf_score_for_rank


def _chunk(chunk_id: str, document: str = "rag.pdf") -> dict:
    return {
        "document": document,
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "page": 1,
        "section": "Retrieval",
    }


class HybridRrfTests(unittest.TestCase):
    def test_same_chunk_in_both_lists_appears_once(self):
        vector_hits = [_chunk("chunk_0"), _chunk("chunk_1")]
        bm25_hits = [_chunk("chunk_1"), _chunk("chunk_2")]

        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], rrf_k=60)
        keys = [make_chunk_key(chunk) for chunk in fused]

        self.assertEqual(len(fused), 3)
        self.assertEqual(len(keys), len(set(keys)))

    def test_chunk_in_both_lists_ranks_higher(self):
        # chunk_1 is in both lists, so its RRF scores add up.
        vector_hits = [_chunk("chunk_0"), _chunk("chunk_1")]
        bm25_hits = [_chunk("chunk_1"), _chunk("chunk_2")]

        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], rrf_k=60)
        fused_ids = [chunk["chunk_id"] for chunk in fused]

        self.assertEqual(fused_ids[0], "chunk_1")
        self.assertEqual(set(fused_ids), {"chunk_0", "chunk_1", "chunk_2"})

        both_lists_score = rrf_score_for_rank(2, 60) + rrf_score_for_rank(1, 60)
        vector_only_score = rrf_score_for_rank(1, 60)
        bm25_only_score = rrf_score_for_rank(2, 60)

        self.assertAlmostEqual(fused[0]["score"], both_lists_score)
        self.assertAlmostEqual(fused[1]["score"], vector_only_score)
        self.assertAlmostEqual(fused[2]["score"], bm25_only_score)

    def test_same_chunk_id_from_different_documents_stays_separate(self):
        list_a = [_chunk("chunk_0", document="a.pdf")]
        list_b = [_chunk("chunk_0", document="b.pdf")]

        fused = reciprocal_rank_fusion([list_a, list_b], rrf_k=60)

        self.assertEqual(len(fused), 2)
        documents = {chunk["document"] for chunk in fused}
        self.assertEqual(documents, {"a.pdf", "b.pdf"})

    def test_empty_lists_return_empty(self):
        self.assertEqual(reciprocal_rank_fusion([[], []], rrf_k=60), [])
        self.assertEqual(reciprocal_rank_fusion([], rrf_k=60), [])


if __name__ == "__main__":
    unittest.main()
