import tempfile
import unittest
from pathlib import Path

from app.core.chunking import chunk_pages, split_text
from app.core.pdf_loader import load_pdf
from app.core.sample_pdf import ensure_sample_pdf


class ChunkingTests(unittest.TestCase):
    def test_split_text_respects_size_and_overlap(self):
        text = ("Retrieval augmented generation. " * 40).strip()
        chunks = split_text(text, chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 + 20 for chunk in chunks))

    def test_chunk_pages_have_ids_and_page_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = ensure_sample_pdf(Path(tmp) / "rag_intro.pdf")
            pages = load_pdf(pdf_path)
            chunks = chunk_pages(pages)
            self.assertEqual(pages[0]["page"], 1)
            self.assertEqual(pages[-1]["page"], 3)
            self.assertTrue(chunks)
            self.assertEqual(chunks[0]["chunk_id"], "chunk_0")
            self.assertTrue({c["page"] for c in chunks} <= {1, 2, 3})
            self.assertTrue(all(c["document"] == "rag_intro.pdf" for c in chunks))
            self.assertTrue(all(c["content_type"] == "text" for c in chunks))


if __name__ == "__main__":
    unittest.main()
