import tempfile
import unittest
from pathlib import Path

import pymupdf

from app.core.chunking import chunk_pages
from app.core.pdf_loader import load_pdf


def _write_survey_like_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 40), "TABLE I")
    page.insert_text((200, 40), "SUMMARY OF RAG METHODS")
    headers = ["Method", "Source", "Granularity", "Process"]
    xs = [80, 180, 300, 430]
    for x, header in zip(xs, headers):
        page.insert_text((x, 70), header)
    rows = [
        ("DenseX", "FactoidWiki", "Proposition", "Once"),
        ("FLARE", "Wikipedia", "Sentence", "Adaptive"),
    ]
    for index, row in enumerate(rows):
        y = 95 + index * 16
        for x, cell in zip(xs, row):
            page.insert_text((x, y), cell)

    fig = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 24, 24), False)
    fig.insert_image(pymupdf.Rect(72, 80, 300, 220), pixmap=pix)
    fig.insert_textbox(
        pymupdf.Rect(72, 240, 520, 300),
        "Fig. 1. Technology tree of RAG research covering pre-training and inference.",
        fontsize=11,
    )
    fig.insert_textbox(
        pymupdf.Rect(72, 320, 520, 400),
        "Naive RAG follows indexing, retrieval, and generation.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return path


class PdfLoaderTests(unittest.TestCase):
    def test_extracts_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = _write_survey_like_pdf(Path(tmp) / "survey.pdf")
            pages = load_pdf(pdf_path)
            self.assertEqual(len(pages), 2)

            tables = pages[0]["tables"]
            self.assertTrue(tables)
            table = tables[0]
            self.assertIn("[Table]", table)
            self.assertIn("DenseX", table)
            self.assertIn("Proposition", table)
            self.assertIn("FLARE", table)

            figures = pages[1]["figures"]
            self.assertTrue(figures)
            self.assertTrue(any("Fig. 1" in item for item in figures))
            self.assertNotIn("Fig. 1", pages[1]["text"])
            self.assertIn("Naive RAG", pages[1]["text"])

            chunks = chunk_pages(pages)
            types = {chunk["content_type"] for chunk in chunks}
            self.assertEqual(types, {"text", "table", "figure"})


if __name__ == "__main__":
    unittest.main()
