from app.config import settings

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0] if separators else ""
    next_separators = separators[1:] if separators else []

    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(separator)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joiner = separator if current else ""
        candidate = f"{current}{joiner}{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) > chunk_size:
            chunks.extend(_split_recursive(piece, next_separators, chunk_size))
            current = ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Recursive character split by size with overlap between adjacent chunks."""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    parts = _split_recursive(text, _SEPARATORS, size)
    if overlap <= 0 or len(parts) <= 1:
        return [p.strip() for p in parts if p.strip()]

    overlapped: list[str] = []
    for index, part in enumerate(parts):
        if index == 0:
            overlapped.append(part)
            continue
        previous = overlapped[-1]
        prefix = previous[-overlap:] if len(previous) > overlap else previous
        combined = prefix + part
        overlapped.append(combined if len(combined) <= size + overlap else part)
    return [p.strip() for p in overlapped if p.strip()]


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split loaded pages into overlapping chunks with chunk_id and page metadata."""
    chunks: list[dict] = []
    chunk_index = 0
    for page in pages:
        for piece in split_text(page.get("text") or ""):
            chunks.append(
                {
                    "text": piece,
                    "page": page["page"],
                    "document": page["document"],
                    "chunk_id": f"chunk_{chunk_index}",
                }
            )
            chunk_index += 1
    return chunks
