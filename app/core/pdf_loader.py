from pathlib import Path

import pymupdf


def load_pdf(path: str | Path) -> list[dict]:
    """Extract page text with basic metadata: page (1-indexed) and document name."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pdf_path.name
    pages: list[dict] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append(
                {
                    "text": text,
                    "page": index + 1,
                    "document": document,
                }
            )
    return pages


def list_pdfs(directory: str | Path) -> list[Path]:
    folder = Path(directory)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())
