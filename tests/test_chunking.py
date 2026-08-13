from app.core.chunking import chunk_pages, split_text
from app.core.pdf_loader import load_pdf
from app.core.sample_pdf import ensure_sample_pdf


def test_split_text_respects_size_and_overlap():
    text = ("Retrieval augmented generation. " * 40).strip()
    chunks = split_text(text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 80 + 20 for c in chunks)


def test_chunk_pages_have_ids_and_page_numbers(tmp_path):
    pdf_path = ensure_sample_pdf(tmp_path / "rag_intro.pdf")
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    assert pages[0]["page"] == 1
    assert pages[-1]["page"] == 3
    assert chunks
    assert chunks[0]["chunk_id"] == "chunk_0"
    assert {c["page"] for c in chunks} <= {1, 2, 3}
    assert all(c["document"] == "rag_intro.pdf" for c in chunks)
