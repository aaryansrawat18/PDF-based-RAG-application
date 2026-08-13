import re

from app.config import settings

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
UNKNOWN_SECTION = "Unknown"

_HEADING_PREFIX = re.compile(
    r"^(?:"
    r"[IVXLC]{1,8}\.\s+"
    r"|[A-Z]\.\s+"
    r"|\d+(?:\.\d+)*\.?\s+"
    r")"
)
_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "via",
    "with",
}


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


def _looks_like_heading(text: str) -> bool:
    """True for short title-case / numbered / ALL-CAPS section headings."""
    line = " ".join((text or "").split())
    if not line or len(line) > 90:
        return False
    if line.startswith("[Table]") or line.startswith("[Figure]"):
        return False
    words = line.split()
    if not 1 <= len(words) <= 12:
        return False
    if line.endswith((".", ",", ";", ":", "!")):
        return False

    stripped = _HEADING_PREFIX.sub("", line).strip() or line
    if stripped.isupper() and any(char.isalpha() for char in stripped):
        return True
    if _HEADING_PREFIX.match(line):
        return True

    significant = 0
    titled = 0
    for word in words:
        token = word.strip("?()[]")
        if not token:
            continue
        if token.lower() in _SMALL_WORDS:
            continue
        significant += 1
        if token[0].isupper() or token[0].isdigit():
            titled += 1
    return significant > 0 and titled == significant


def _section_title(text: str) -> str:
    line = " ".join((text or "").split())
    stripped = _HEADING_PREFIX.sub("", line).strip()
    return stripped or UNKNOWN_SECTION


def _section_groups(text: str, current_section: str) -> tuple[list[tuple[str, str]], str]:
    """Split page text on headings; carry the last section forward."""
    blocks = [block.strip() for block in (text or "").split("\n\n") if block.strip()]
    if not blocks:
        return [], current_section

    groups: list[tuple[str, list[str]]] = []
    section = current_section
    bucket: list[str] = []

    def flush() -> None:
        if bucket:
            groups.append((section, list(bucket)))
            bucket.clear()

    for block in blocks:
        if _looks_like_heading(block):
            flush()
            section = _section_title(block)
            bucket.append(block)
            continue
        bucket.append(block)
    flush()
    return [(name, "\n\n".join(parts)) for name, parts in groups], section


def _make_chunk(
    text: str,
    page: dict,
    chunk_index: int,
    content_type: str,
    section: str,
) -> dict:
    return {
        "text": text,
        "page": page["page"],
        "section": section or UNKNOWN_SECTION,
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
    current_section = UNKNOWN_SECTION
    for page in pages:
        groups, current_section = _section_groups(page.get("text") or "", current_section)
        for section, text in groups:
            for piece in split_text(text):
                chunks.append(_make_chunk(piece, page, chunk_index, "text", section))
                chunk_index += 1
        for table in page.get("tables") or []:
            if table.strip():
                for piece in _split_labeled_block(table, size, overlap):
                    chunks.append(
                        _make_chunk(piece, page, chunk_index, "table", current_section)
                    )
                    chunk_index += 1
        for figure in page.get("figures") or []:
            if figure.strip():
                for piece in _split_labeled_block(figure, size, overlap):
                    chunks.append(
                        _make_chunk(piece, page, chunk_index, "figure", current_section)
                    )
                    chunk_index += 1
    return chunks
