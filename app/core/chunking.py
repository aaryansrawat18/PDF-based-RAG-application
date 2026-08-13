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


def _split_labeled_block(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Keep table/figure headers and split labeled rows to embedding-friendly size."""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(text) <= size:
        return [text.strip()]

    prefix_lines: list[str] = []
    rows: list[str] = []
    for line in lines:
        if line.startswith("- ") and prefix_lines:
            rows.append(line)
        elif not rows:
            prefix_lines.append(line)
        else:
            rows.append(line)
    prefix = "\n".join(prefix_lines).strip()
    if not rows:
        return split_text(text, chunk_size=size, chunk_overlap=overlap)

    parts: list[str] = []
    current: list[str] = []
    current_len = len(prefix)
    for row in rows:
        extra = len(row) + 1
        if current and current_len + extra > size:
            parts.append(prefix + "\n" + "\n".join(current))
            overlap_rows = current[-1:] if overlap > 0 else []
            current = overlap_rows + [row]
            current_len = len(prefix) + sum(len(item) + 1 for item in current)
        else:
            current.append(row)
            current_len += extra
    if current:
        parts.append(prefix + "\n" + "\n".join(current))
    return [part.strip() for part in parts if part.strip()]


def _make_chunk(text: str, page: dict, chunk_index: int, content_type: str) -> dict:
    return {
        "text": text,
        "page": page["page"],
        "document": page["document"],
        "chunk_id": f"chunk_{chunk_index}",
        "content_type": content_type,
    }


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split body text; keep tables/figures intact except when they exceed chunk size."""
    chunks: list[dict] = []
    chunk_index = 0
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    for page in pages:
        for piece in split_text(page.get("text") or ""):
            chunks.append(_make_chunk(piece, page, chunk_index, "text"))
            chunk_index += 1
        for table in page.get("tables") or []:
            if table.strip():
                for piece in _split_labeled_block(table, size, overlap):
                    chunks.append(_make_chunk(piece, page, chunk_index, "table"))
                    chunk_index += 1
        for figure in page.get("figures") or []:
            if figure.strip():
                for piece in _split_labeled_block(figure, size, overlap):
                    chunks.append(_make_chunk(piece, page, chunk_index, "figure"))
                    chunk_index += 1
    return chunks
