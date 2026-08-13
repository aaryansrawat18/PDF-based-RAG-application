import unittest
from unittest.mock import patch

from app.config import settings
from app.core.rag_chain import run_rag
from app.graph.nodes import (
    active_query,
    answer_is_ok,
    context_is_ok,
    quality_check_node,
    retrieve_node,
    rewrite_query_node,
)
from app.graph.pipeline import (
    build_graph,
    reset_graph,
    route_after_generate,
    route_after_quality,
)


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


WEAK = _chunk("weak", "Calendar leftovers and unrelated footnotes.", 0.1)
STRONG = _chunk(
    "strong",
    "Retrieval augmented generation combines search with an LLM.",
    3.2,
)


class QualityHelperTests(unittest.TestCase):
    def test_empty_pruned_is_not_ok(self):
        self.assertFalse(context_is_ok([]))
        self.assertFalse(context_is_ok(None))

    def test_low_score_is_not_ok(self):
        self.assertFalse(context_is_ok([WEAK]))

    def test_good_context_is_ok(self):
        self.assertTrue(context_is_ok([STRONG]))

    def test_blank_text_is_not_ok(self):
        self.assertFalse(context_is_ok([_chunk("empty", "   ", 4.0)]))

    def test_quality_check_node_sets_flag(self):
        self.assertEqual(
            quality_check_node({"pruned": [STRONG]}),
            {"context_ok": True},
        )
        self.assertEqual(
            quality_check_node({"pruned": [WEAK]}),
            {"context_ok": False},
        )

    def test_refusal_answer_is_not_ok(self):
        self.assertFalse(answer_is_ok(""))
        self.assertFalse(answer_is_ok("I don't know."))
        self.assertFalse(answer_is_ok("The context does not contain the answer."))

    def test_grounded_answer_is_ok(self):
        self.assertTrue(answer_is_ok("RAG retrieves passages then generates."))

    def test_active_query_prefers_rewrite(self):
        self.assertEqual(
            active_query(
                {
                    "question": "What is it?",
                    "rewritten_query": "retrieval augmented generation",
                }
            ),
            "retrieval augmented generation",
        )
        self.assertEqual(active_query({"question": "What is RAG?"}), "What is RAG?")


class RouteTests(unittest.TestCase):
    def test_bad_context_routes_to_rewrite(self):
        self.assertEqual(
            route_after_quality({"context_ok": False, "retry_count": 0}),
            "rewrite",
        )

    def test_good_context_routes_to_generate(self):
        self.assertEqual(
            route_after_quality({"context_ok": True, "retry_count": 0}),
            "generate",
        )

    def test_exhausted_retrieve_retries_route_to_generate(self):
        self.assertEqual(
            route_after_quality(
                {
                    "context_ok": False,
                    "retry_count": settings.max_retrieve_retries,
                }
            ),
            "generate",
        )

    def test_weak_answer_routes_to_retry(self):
        self.assertEqual(
            route_after_generate(
                {
                    "answer": "I don't know.",
                    "pruned": [STRONG],
                    "generate_retry_count": 1,
                }
            ),
            "retry",
        )

    def test_weak_answer_stops_after_max_generate_retries(self):
        self.assertEqual(
            route_after_generate(
                {
                    "answer": "I don't know.",
                    "pruned": [STRONG],
                    "generate_retry_count": settings.max_generate_retries + 1,
                }
            ),
            "end",
        )

    def test_empty_context_does_not_retry_generate(self):
        self.assertEqual(
            route_after_generate(
                {
                    "answer": "I don't know.",
                    "pruned": [],
                    "generate_retry_count": 1,
                }
            ),
            "end",
        )


class NodeTests(unittest.TestCase):
    @patch("app.graph.nodes.reciprocal_rank_fusion", return_value=[STRONG])
    @patch("app.graph.nodes.search_bm25", return_value=[])
    @patch("app.graph.nodes.similarity_search", return_value=[])
    @patch("app.graph.nodes.embed_query", return_value=[0.1, 0.2])
    def test_retrieve_uses_rewritten_query(
        self,
        mock_embed,
        mock_vector,
        mock_bm25,
        mock_rrf,
    ):
        retrieve_node(
            {
                "question": "What is it?",
                "rewritten_query": "retrieval augmented generation definition",
            }
        )
        mock_embed.assert_called_once_with("retrieval augmented generation definition")
        mock_bm25.assert_called_once()
        self.assertEqual(
            mock_bm25.call_args.args[0],
            "retrieval augmented generation definition",
        )
        mock_vector.assert_called_once()
        mock_rrf.assert_called_once()

    @patch("app.graph.nodes.generate")
    def test_rewrite_query_node_stores_cleaned_query(self, mock_generate):
        mock_generate.return_value = 'Rewritten query: "DenseX retrieval granularity"'
        output = rewrite_query_node(
            {
                "question": "What granularity?",
                "pruned": [WEAK],
                "retry_count": 0,
            }
        )
        self.assertEqual(output["rewritten_query"], "DenseX retrieval granularity")
        self.assertEqual(output["retry_count"], 1)
        self.assertFalse(output["context_ok"])
        mock_generate.assert_called_once()
        self.assertEqual(
            mock_generate.call_args.kwargs["model"],
            settings.rewrite_model,
        )


class GraphLoopTests(unittest.TestCase):
    def tearDown(self):
        reset_graph()

    def test_graph_includes_decision_nodes(self):
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

    def test_weak_first_retrieval_rewrites_and_retrieves_again(self):
        retrieve_calls: list[str] = []

        def fake_retrieve(state):
            query = state.get("rewritten_query") or state["question"]
            retrieve_calls.append(query)
            if state.get("rewritten_query"):
                return {"retrieved": [STRONG]}
            return {"retrieved": [WEAK]}

        def fake_rerank(state):
            return {"reranked": list(state.get("retrieved") or [])}

        def fake_prune(state):
            return {"pruned": list(state.get("reranked") or [])}

        def fake_generate(state):
            return {
                "answer": "RAG combines retrieval with generation.",
                "sources": [{"chunk_id": "strong", "page": 1}],
                "generate_retry_count": int(state.get("generate_retry_count") or 0) + 1,
            }

        with (
            patch("app.graph.pipeline.retrieve_node", side_effect=fake_retrieve),
            patch("app.graph.pipeline.rerank_node", side_effect=fake_rerank),
            patch("app.graph.pipeline.prune_node", side_effect=fake_prune),
            patch("app.graph.pipeline.generate_node", side_effect=fake_generate),
            patch(
                "app.graph.nodes.generate",
                return_value="retrieval augmented generation definition",
            ),
        ):
            reset_graph()
            result = build_graph().invoke(
                {
                    "question": "What is it?",
                    "retry_count": 0,
                    "generate_retry_count": 0,
                    "rewritten_query": "",
                    "context_ok": False,
                }
            )

        self.assertGreaterEqual(len(retrieve_calls), 2)
        self.assertEqual(retrieve_calls[0], "What is it?")
        self.assertEqual(
            retrieve_calls[1],
            "retrieval augmented generation definition",
        )
        self.assertTrue(result["context_ok"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["answer"], "RAG combines retrieval with generation.")

    def test_refusal_answer_triggers_one_generate_retry(self):
        generate_calls: list[str] = []

        def fake_retrieve(_state):
            return {"retrieved": [STRONG]}

        def fake_rerank(state):
            return {"reranked": list(state.get("retrieved") or [])}

        def fake_prune(state):
            return {"pruned": list(state.get("reranked") or [])}

        def fake_generate(state):
            attempts = int(state.get("generate_retry_count") or 0) + 1
            if attempts == 1:
                generate_calls.append("refuse")
                return {
                    "answer": "I don't know.",
                    "sources": [],
                    "generate_retry_count": attempts,
                }
            generate_calls.append("retry")
            return {
                "answer": "DenseX uses proposition granularity.",
                "sources": [{"chunk_id": "strong"}],
                "generate_retry_count": attempts,
            }

        with (
            patch("app.graph.pipeline.retrieve_node", side_effect=fake_retrieve),
            patch("app.graph.pipeline.rerank_node", side_effect=fake_rerank),
            patch("app.graph.pipeline.prune_node", side_effect=fake_prune),
            patch("app.graph.pipeline.generate_node", side_effect=fake_generate),
        ):
            reset_graph()
            result = build_graph().invoke(
                {
                    "question": "What granularity does DenseX use?",
                    "retry_count": 0,
                    "generate_retry_count": 0,
                    "rewritten_query": "",
                    "context_ok": False,
                }
            )

        self.assertEqual(generate_calls, ["refuse", "retry"])
        self.assertEqual(result["answer"], "DenseX uses proposition granularity.")
        self.assertEqual(result["generate_retry_count"], 2)

    def test_retrieve_retry_cap_stops_rewrite_loop(self):
        retrieve_calls = {"n": 0}

        def fake_retrieve(_state):
            retrieve_calls["n"] += 1
            return {"retrieved": [WEAK]}

        def fake_generate(state):
            return {
                "answer": "I don't know.",
                "sources": [],
                "generate_retry_count": int(state.get("generate_retry_count") or 0) + 1,
            }

        with (
            patch("app.graph.pipeline.retrieve_node", side_effect=fake_retrieve),
            patch(
                "app.graph.pipeline.rerank_node",
                side_effect=lambda state: {"reranked": list(state.get("retrieved") or [])},
            ),
            patch(
                "app.graph.pipeline.prune_node",
                side_effect=lambda state: {"pruned": list(state.get("reranked") or [])},
            ),
            patch("app.graph.pipeline.generate_node", side_effect=fake_generate),
            patch("app.graph.nodes.generate", return_value="another rewrite"),
        ):
            reset_graph()
            result = build_graph().invoke(
                {
                    "question": "zzzz not in corpus",
                    "retry_count": 0,
                    "generate_retry_count": 0,
                    "rewritten_query": "",
                    "context_ok": False,
                }
            )

        self.assertEqual(retrieve_calls["n"], settings.max_retrieve_retries + 1)
        self.assertEqual(result["retry_count"], settings.max_retrieve_retries)
        self.assertFalse(result["context_ok"])


class RunRagTests(unittest.TestCase):
    @patch("app.core.rag_chain.get_graph")
    def test_run_rag_includes_latency_ms(self, mock_get_graph):
        mock_get_graph.return_value.invoke.return_value = {
            "answer": "RAG is retrieval augmented generation.",
            "sources": [],
            "retry_count": 0,
        }
        result = run_rag("What is RAG?")
        self.assertIn("latency_ms", result)
        self.assertIsInstance(result["latency_ms"], int)
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertEqual(result["answer"], "RAG is retrieval augmented generation.")
        payload = mock_get_graph.return_value.invoke.call_args.args[0]
        self.assertEqual(payload["retry_count"], 0)
        self.assertEqual(payload["generate_retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
