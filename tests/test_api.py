import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body, {"status": "ok"})

    def test_ask_rejects_empty_question(self):
        response = self.client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("question", response.json()["detail"])

    @patch("app.api.routes.run_rag")
    def test_ask_response_shape(self, mock_run_rag):
        mock_run_rag.return_value = {
            "answer": "RAG is retrieval augmented generation.",
            "sources": [
                {
                    "page": 3,
                    "document": "rag.pdf",
                    "section": "Introduction",
                    "chunk_id": "chunk_0",
                    "content_type": "text",
                    "score": 0.91,
                }
            ],
            "latency_ms": 1420,
        }

        response = self.client.post("/ask", json={"question": "What is RAG?"})

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertIn("answer", body)
        self.assertIn("sources", body)
        self.assertIn("latency_ms", body)
        self.assertEqual(body["latency_ms"], 1420)
        self.assertIsInstance(body["answer"], str)
        self.assertTrue(body["answer"])
        self.assertIsInstance(body["sources"], list)
        self.assertEqual(len(body["sources"]), 1)

        source = body["sources"][0]
        self.assertEqual(source["page"], 3)
        self.assertEqual(source["document"], "rag.pdf")
        self.assertEqual(source["section"], "Introduction")
        self.assertEqual(source["chunk_id"], "chunk_0")
        self.assertEqual(source["score"], 0.91)

        mock_run_rag.assert_called_once_with("What is RAG?")

    @patch("app.api.routes.ingest_pdfs")
    def test_ingest_response_shape(self, mock_ingest_pdfs):
        mock_ingest_pdfs.return_value = [
            {
                "document": "rag.pdf",
                "pages": 3,
                "chunks": 12,
                "tables": 1,
                "figures": 2,
            }
        ]

        response = self.client.post(
            "/ingest",
            json={"pdf_path": "data/source_pdfs/rag.pdf"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("message", body)
        self.assertIn("documents", body)
        self.assertEqual(len(body["documents"]), 1)
        self.assertEqual(body["documents"][0]["document"], "rag.pdf")
        self.assertEqual(body["documents"][0]["pages"], 3)
        self.assertEqual(body["documents"][0]["chunks"], 12)

        mock_ingest_pdfs.assert_called_once_with("data/source_pdfs/rag.pdf")

    @patch("app.api.routes.ingest_pdfs")
    def test_ingest_missing_pdf_returns_404(self, mock_ingest_pdfs):
        mock_ingest_pdfs.side_effect = FileNotFoundError("PDF not found: missing.pdf")
        response = self.client.post("/ingest", json={"pdf_path": "missing.pdf"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
