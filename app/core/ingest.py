from pathlib import Path

from app.config import settings
from app.core.bm25 import update_bm25_corpus
from app.core.chunking import chunk_pages
from app.core.embeddings import embed_documents
from app.core.pdf_loader import list_pdfs, load_pdf
from app.core.sample_pdf import ensure_sample_pdf
from app.core.vectorstore import upsert_chunks


def ingest_pdf(pdf_path: str | Path) -> dict:
    path = Path(pdf_path)
    pages = load_pdf(path)
    chunks = chunk_pages(pages)
    embeddings = embed_documents([chunk["text"] for chunk in chunks])
    stored = upsert_chunks(chunks, embeddings)
    update_bm25_corpus(path.name, chunks)
    return {
        "document": path.name,
        "pages": len(pages),
        "chunks": stored,
        "tables": sum(len(page.get("tables") or []) for page in pages),
        "figures": sum(len(page.get("figures") or []) for page in pages),
    }


def ingest_pdfs(pdf_path: str | Path | None = None) -> list[dict]:
    if pdf_path:
        return [ingest_pdf(pdf_path)]

    pdfs = list_pdfs(settings.source_pdfs_dir)
    if not pdfs:
        sample = ensure_sample_pdf(Path(settings.source_pdfs_dir) / "rag_intro.pdf")
        pdfs = [sample]
    return [ingest_pdf(path) for path in pdfs]
