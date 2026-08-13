import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.rag_chain import run_rag
from app.eval.dataset import (
    examples_to_langsmith,
    load_eval_dataset,
    parse_example,
)
from app.eval.evaluators import (
    aggregate_scores,
    comparison_table,
    faithfulness,
    format_comparison_table,
    hit_rate,
    mrr,
    precision,
    recall,
    score_prediction,
    token_f1,
)
from app.graph.pipeline import (
    build_baseline_graph,
    build_graph,
    reset_graph,
)


class DatasetTests(unittest.TestCase):
    def test_shipped_jsonl_has_required_fields(self):
        examples = load_eval_dataset()
        self.assertGreaterEqual(len(examples), 20)
        self.assertLessEqual(len(examples), 40)
        ids = [row["id"] for row in examples]
        self.assertEqual(len(ids), len(set(ids)))
        for row in examples:
            self.assertTrue(row["question"])
            self.assertTrue(row["reference_answer"])
            self.assertTrue(row["expected_pages"])
            self.assertTrue(all(isinstance(page, int) and page >= 1 for page in row["expected_pages"]))

    def test_parse_example_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            parse_example({"question": "What is RAG?"}, line_no=1)

    def test_load_eval_dataset_reads_temp_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "t1",
                        "question": "What is RAG?",
                        "reference_answer": "Retrieval then generation.",
                        "expected_pages": [3],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_eval_dataset(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["expected_pages"], [3])

    @patch("langsmith.Client")
    def test_upload_creates_dataset_when_missing(self, mock_client_cls):
        from app.eval.dataset import upload_langsmith_dataset

        client = mock_client_cls.return_value
        client.has_dataset.return_value = False
        client.create_dataset.return_value.id = "ds-1"
        client.create_dataset.return_value.name = "rag-eval"
        examples = [
            {
                "id": "q01",
                "question": "What is RAG?",
                "reference_answer": "Retrieval then generation.",
                "expected_pages": [1],
            }
        ]
        result = upload_langsmith_dataset(examples)
        client.create_dataset.assert_called_once()
        client.create_examples.assert_called_once()
        self.assertTrue(result["created"])
        self.assertEqual(result["count"], 1)

    def test_langsmith_payload_splits_inputs_and_outputs(self):
        payload = examples_to_langsmith(
            [
                {
                    "id": "q01",
                    "question": "What is RAG?",
                    "reference_answer": "Retrieval then generation.",
                    "expected_pages": [1, 3],
                }
            ]
        )
        self.assertEqual(payload[0]["inputs"]["question"], "What is RAG?")
        self.assertEqual(payload[0]["outputs"]["expected_pages"], [1, 3])
        self.assertEqual(payload[0]["metadata"]["id"], "q01")


class EvaluatorTests(unittest.TestCase):
    def test_hit_recall_precision_mrr(self):
        predicted = [10, 3, 7]
        expected = [3, 4]
        self.assertEqual(hit_rate(predicted, expected), 1.0)
        self.assertEqual(recall(predicted, expected), 0.5)
        self.assertAlmostEqual(precision(predicted, expected), 1 / 3)
        self.assertAlmostEqual(mrr(predicted, expected), 0.5)

    def test_miss_is_zero(self):
        self.assertEqual(hit_rate([1, 2], [9]), 0.0)
        self.assertEqual(recall([1, 2], [9]), 0.0)
        self.assertEqual(mrr([1, 2], [9]), 0.0)
        self.assertEqual(precision([], [1]), 0.0)

    def test_token_f1_and_faithfulness(self):
        self.assertGreater(
            token_f1(
                "DenseX uses proposition granularity (page 6).",
                "DenseX uses proposition granularity.",
            ),
            0.7,
        )
        self.assertGreater(
            faithfulness(
                "DenseX uses proposition granularity.",
                ["DenseX uses proposition as the retrieval unit."],
            ),
            0.5,
        )
        self.assertEqual(
            faithfulness("invented jargon xyzzy", ["unrelated footnotes"]),
            0.0,
        )

    def test_score_prediction_uses_retrieved_and_source_pages(self):
        example = {
            "id": "q13",
            "question": "What retrieval granularity does DenseX use?",
            "reference_answer": "DenseX uses proposition granularity.",
            "expected_pages": [6],
        }
        result = {
            "pipeline": "advanced",
            "answer": "DenseX uses proposition granularity (page 6).",
            "retrieved_pages": [2, 6, 8],
            "source_pages": [6],
            "pruned": [{"text": "DenseX uses proposition granularity at inference.", "page": 6}],
            "latency_ms": 1200,
            "context_tokens": 80,
            "answer_tokens": 12,
        }
        scored = score_prediction(example, result)
        self.assertEqual(scored["hit_rate"], 1.0)
        self.assertEqual(scored["mrr"], 0.5)
        self.assertEqual(scored["context_hit_rate"], 1.0)
        self.assertGreater(scored["relevance"], 0.4)
        self.assertGreater(scored["faithfulness"], 0.5)

    def test_aggregate_and_comparison_table(self):
        baseline = [
            {
                "hit_rate": 0.5,
                "recall": 0.4,
                "precision": 0.2,
                "mrr": 0.3,
                "context_hit_rate": 0.5,
                "context_recall": 0.4,
                "faithfulness": 0.6,
                "relevance": 0.5,
                "latency_ms": 1000,
                "context_tokens": 4000,
                "answer_tokens": 80,
            }
        ]
        advanced = [
            {
                "hit_rate": 1.0,
                "recall": 0.8,
                "precision": 0.4,
                "mrr": 0.7,
                "context_hit_rate": 1.0,
                "context_recall": 0.8,
                "faithfulness": 0.8,
                "relevance": 0.7,
                "latency_ms": 1400,
                "context_tokens": 900,
                "answer_tokens": 70,
            }
        ]
        table = comparison_table(
            {
                "baseline": aggregate_scores(baseline),
                "advanced": aggregate_scores(advanced),
            }
        )
        by_metric = {row["metric"]: row for row in table}
        self.assertGreater(by_metric["hit_rate"]["delta"], 0)
        self.assertLess(by_metric["context_tokens"]["delta"], 0)
        text = format_comparison_table(table)
        self.assertIn("hit_rate", text)
        self.assertIn("baseline", text)


class BaselineGraphTests(unittest.TestCase):
    def tearDown(self):
        reset_graph()

    def test_baseline_graph_is_retrieve_then_generate(self):
        node_ids = set(build_baseline_graph().get_graph().nodes)
        self.assertIn("retrieve", node_ids)
        self.assertIn("generate", node_ids)
        self.assertNotIn("rerank", node_ids)
        self.assertNotIn("prune", node_ids)
        self.assertNotIn("rewrite_query", node_ids)

    def test_advanced_graph_keeps_decision_nodes(self):
        node_ids = set(build_graph().get_graph().nodes)
        for name in ("retrieve", "rerank", "prune", "quality_check", "rewrite_query", "generate"):
            self.assertIn(name, node_ids)

    @patch("app.graph.nodes.similarity_search")
    @patch("app.graph.nodes.embed_query", return_value=[0.1, 0.2])
    def test_baseline_retrieve_skips_bm25(self, _embed, mock_search):
        from app.graph.nodes import baseline_retrieve_node

        mock_search.return_value = [
            {
                "text": "RAG retrieves then generates.",
                "page": 3,
                "chunk_id": "c0",
                "document": "Document.pdf",
            }
        ]
        with patch("app.graph.nodes.search_bm25") as mock_bm25:
            output = baseline_retrieve_node({"question": "What is RAG?"})
            mock_bm25.assert_not_called()
        self.assertEqual(output["retrieved"][0]["page"], 3)
        self.assertEqual(output["pruned"][0]["page"], 3)
        self.assertTrue(output["context_ok"])

    @patch("app.core.rag_chain.get_baseline_graph")
    @patch("app.core.rag_chain.get_graph")
    def test_run_rag_baseline_uses_baseline_graph(self, mock_advanced, mock_baseline):
        mock_baseline.return_value.invoke.return_value = {
            "answer": "Naive RAG is retrieve-read.",
            "sources": [{"page": 3}],
            "retrieved": [{"page": 3, "text": "Retrieve-Read"}],
            "pruned": [{"page": 3, "text": "Retrieve-Read"}],
        }
        result = run_rag("What is Naive RAG?", pipeline="baseline")
        mock_baseline.assert_called_once()
        mock_advanced.assert_not_called()
        self.assertEqual(result["pipeline"], "baseline")
        self.assertEqual(result["retrieved_pages"], [3])
        self.assertIn("context_tokens", result)
        self.assertIn("latency_ms", result)

    def test_run_rag_rejects_unknown_pipeline(self):
        with self.assertRaises(ValueError):
            run_rag("What is RAG?", pipeline="semantic-cache")


if __name__ == "__main__":
    unittest.main()
