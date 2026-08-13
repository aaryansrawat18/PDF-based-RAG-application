import unittest

from unittest.mock import patch

from app.core.reranker import rerank
from app.graph.nodes import prune_node, rerank_node
from app.graph.pipeline import build_graph


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "document": "rag.pdf",
        "chunk_id": chunk_id,
        "text": text,
        "page": 1,
        "section": "Retrieval",
        "content_type": "text",
        "score": 0.01,
    }


def _score_by_keyword(query: str, texts: list[str]) -> list[float]:
    """Higher score when the query token appears later in the text list.

    Used so the fused order (A, B, C) is not the reranked order.
    """
    needle = query.lower()
    scores = []
    for text in texts:
        if needle in text.lower():
            scores.append(10.0 + text.lower().index(needle) * 0.01)
        else:
            scores.append(0.1)
    return scores


class RerankerTests(unittest.TestCase):
    def test_rerank_changes_order(self):
        chunks = [
            _chunk("chunk_a", "Hybrid search fuses dense and sparse hits."),
            _chunk("chunk_b", "Prompt caching is a later phase."),
            _chunk("chunk_c", "Cross-encoder rerank scores query and chunk together."),
        ]

        reranked = rerank(
            "rerank",
            chunks,
            top_k=10,
            score_fn=_score_by_keyword,
        )

        fused_ids = [chunk["chunk_id"] for chunk in chunks]
        reranked_ids = [chunk["chunk_id"] for chunk in reranked]

        self.assertEqual(fused_ids, ["chunk_a", "chunk_b", "chunk_c"])
        self.assertEqual(reranked_ids[0], "chunk_c")
        self.assertNotEqual(reranked_ids, fused_ids)
        self.assertGreater(reranked[0]["score"], reranked[1]["score"])

    def test_rerank_keeps_top_k(self):
        chunks = [
            _chunk("chunk_a", "alpha"),
            _chunk("chunk_b", "beta rerank"),
            _chunk("chunk_c", "gamma"),
        ]

        reranked = rerank(
            "rerank",
            chunks,
            top_k=1,
            score_fn=_score_by_keyword,
        )

        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0]["chunk_id"], "chunk_b")

    def test_empty_chunks_return_empty(self):
        self.assertEqual(rerank("query", [], score_fn=_score_by_keyword), [])

    def test_score_count_mismatch_raises(self):
        chunks = [_chunk("chunk_a", "alpha")]
        with self.assertRaises(ValueError):
            rerank("query", chunks, score_fn=lambda _q, _t: [1.0, 2.0])

    @patch("app.graph.nodes.rerank")
    def test_rerank_node_stores_reranked(self, mock_rerank):
        mock_rerank.return_value = [_chunk("chunk_c", "Cross-encoder hit")]
        output = rerank_node(
            {
                "question": "rerank",
                "retrieved": [_chunk("chunk_a", "Hybrid search")],
            }
        )
        self.assertEqual(output["reranked"][0]["chunk_id"], "chunk_c")
        mock_rerank.assert_called_once()

    def test_prune_node_stores_pruned(self):
        output = prune_node(
            {
                "reranked": [
                    _chunk("keep", "Relevant retrieval passage about RAG."),
                    _chunk("dup", "Relevant retrieval passage about RAG."),
                ]
            }
        )
        # scores default to 0.01 in _chunk; threshold is 0.0 so both pass
        # overlap, then prune keeps the first only
        ids = [chunk["chunk_id"] for chunk in output["pruned"]]
        self.assertEqual(ids, ["keep"])

    def test_graph_has_phase6_decision_nodes(self):
        node_ids = set(build_graph().get_graph().nodes)
        for name in (
            "retrieve",
            "rerank",
            "prune",
            "quality_check",
            "rewrite_query",
            "generate",
        ):
            self.assertIn(name, node_ids)


if __name__ == "__main__":
    unittest.main()
