"""PDF loading with PyMuPDF (ingest step 1).

Returns one dict per page: body text, markdown tables, figure captions,
plus page number and document name for later citations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

_TABLE_TITLE = re.compile(r"^\s*TABLE\s+([IVXLC]+|\d+)\b", re.I)
_FIG_CAPTION = re.compile(r"^\s*Fig(?:ure)?\.?\s*\d+", re.I)
_PAGE_ONLY = re.compile(r"^\s*\d+\s*$")
_CITATION = re.compile(r"\[\d{1,4}\]")


def load_pdf(path: str | Path) -> list[dict]:
    """Extract body text, markdown tables, and figure captions per page."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pdf_path.name
    pages: list[dict] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            tables, table_rects = _extract_tables(page)
            figures, figure_rects = _extract_figures(page)
            skip = table_rects + figure_rects
            text = _extract_body_text(page, skip)
            pages.append(
                {
                    "text": text,
                    "tables": tables,
                    "figures": figures,
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


def _normalize_block(text: str) -> str:
    text = (text or "").replace("\r", "")
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n+", " ", text)
    return re.sub(r" +", " ", text).strip()


def _overlaps(block_rect: pymupdf.Rect, regions: list[pymupdf.Rect], min_frac: float = 0.35) -> bool:
    area = block_rect.width * block_rect.height
    if area <= 0:
        return False
    for region in regions:
        inter = block_rect & region
        if inter.is_empty:
            continue
        if (inter.width * inter.height) / area >= min_frac:
            return True
    return False


def _extract_body_text(page: pymupdf.Page, skip_rects: list[pymupdf.Rect]) -> str:
    parts: list[str] = []
    for block in page.get_text("blocks"):
        rect = pymupdf.Rect(block[:4])
        raw = (block[4] or "").strip()
        if not raw or _PAGE_ONLY.match(raw):
            continue
        if _TABLE_TITLE.match(raw) or _FIG_CAPTION.match(raw):
            continue
        if skip_rects and _overlaps(rect, skip_rects):
            continue
        cleaned = _normalize_block(raw)
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts)


def _extract_figures(page: pymupdf.Page) -> tuple[list[str], list[pymupdf.Rect]]:
    figures: list[str] = []
    rects: list[pymupdf.Rect] = []
    for block in page.get_text("blocks"):
        raw = (block[4] or "").strip()
        if not _FIG_CAPTION.match(raw):
            continue
        caption = _normalize_block(raw)
        if not caption:
            continue
        figures.append(f"[Figure] {caption}")
        rects.append(pymupdf.Rect(block[:4]))

    if not figures:
        for img in page.get_images(full=True):
            image_rects = page.get_image_rects(img[0])
            if image_rects:
                rects.extend(image_rects)
                figures.append(
                    f"[Figure] Diagram on this page "
                    f"({int(image_rects[0].width)}x{int(image_rects[0].height)} pt)."
                )
    return figures, rects


def _extract_tables(page: pymupdf.Page) -> tuple[list[str], list[pymupdf.Rect]]:
    title_block = _table_title_block(page)
    grid_table, grid_rect = _table_from_words(page, title_block)
    if grid_table:
        return [grid_table], [grid_rect] if grid_rect else []

    # Line-based finder only. strategy="text" false-positives on paragraphs.
    return _tables_from_finder(page)


def _table_title_block(page: pymupdf.Page) -> tuple[str, pymupdf.Rect] | None:
    for block in page.get_text("blocks"):
        raw = (block[4] or "").strip()
        if not raw:
            continue
        first = raw.split("\n", 1)[0]
        if _TABLE_TITLE.match(first):
            return _normalize_block(raw), pymupdf.Rect(block[:4])
    return None


def _tables_from_finder(page: pymupdf.Page) -> tuple[list[str], list[pymupdf.Rect]]:
    markdowns: list[str] = []
    rects: list[pymupdf.Rect] = []
    try:
        finder = page.find_tables()
    except Exception:
        return [], []
    for table in finder.tables:
        rows = table.extract() or []
        if not _usable_matrix(rows):
            continue
        markdowns.append(_matrix_to_rag_text("Table", rows))
        rects.append(pymupdf.Rect(table.bbox))
    return markdowns, rects


def _usable_matrix(rows: list[list]) -> bool:
    if len(rows) < 3:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    nonempty = [" ".join(str(c).split()) for row in rows for c in row if c and str(c).strip()]
    if len(nonempty) < width * 2:
        return False
    avg = sum(len(c) for c in nonempty) / len(nonempty)
    return 1 <= avg <= 80


def _table_from_words(
    page: pymupdf.Page,
    title_block: tuple[str, pymupdf.Rect] | None,
) -> tuple[str | None, pymupdf.Rect | None]:
    if not title_block:
        return None, None
    title, title_rect = title_block
    words = [
        w
        for w in page.get_text("words")
        if w[1] >= title_rect.y1 - 1 and not _PAGE_ONLY.match(w[4])
    ]
    if len(words) < 8:
        return None, None

    citation_y = min((w[1] for w in words if _CITATION.search(w[4])), default=None)
    split_y = _header_data_split_y(words)
    if citation_y is not None and citation_y > split_y:
        split_y = min(split_y, citation_y - 2)

    header_words = [w for w in words if w[1] < split_y]
    data_words = [w for w in words if w[1] >= split_y]
    if len(header_words) < 2 or len(data_words) < 4:
        return None, None

    header_clusters, means = _cluster_by_x(header_words)
    if len(means) < 2:
        return None, None
    headers = [" ".join(w[4] for w in sorted(cluster, key=lambda t: (t[1], t[0]))) for cluster in header_clusters]

    row_groups = _group_rows(data_words)
    raw_rows = [_assign_row(group, means) for group in row_groups]
    merged = _merge_wrapped_rows(raw_rows)
    merged = [row for row in merged if any(cell.strip() for cell in row)]
    if len(merged) < 1:
        return None, None

    matrix = [headers, *merged]
    text = _matrix_to_rag_text(title, matrix)
    x0 = min(w[0] for w in words)
    y0 = title_rect.y0
    x1 = max(w[2] for w in words)
    y1 = max(w[3] for w in words)
    return text, pymupdf.Rect(x0, y0, x1, y1)


def _header_data_split_y(words: list, min_gap: float = 8.0) -> float:
    groups = _group_rows(words, y_tol=3.0)
    if not groups:
        return 0.0
    if len(groups) == 1:
        return groups[0][0][1] + 8
    ys = [group[0][1] for group in groups]
    scan = min(5, len(ys) - 1)
    for index in range(scan):
        gap = ys[index + 1] - ys[index]
        if gap >= min_gap:
            return (ys[index] + ys[index + 1]) / 2
    return (ys[0] + ys[1]) / 2


def _cluster_by_x(words: list, thresh: float = 42.0) -> tuple[list[list], list[float]]:
    clusters: list[list] = []
    means: list[float] = []
    for word in sorted(words, key=lambda item: item[0]):
        x = (word[0] + word[2]) / 2
        best_i = None
        best_d = None
        for index, mean in enumerate(means):
            dist = abs(x - mean)
            if dist < thresh and (best_d is None or dist < best_d):
                best_i = index
                best_d = dist
        if best_i is None:
            clusters.append([word])
            means.append(x)
            continue
        clusters[best_i].append(word)
        xs = [(item[0] + item[2]) / 2 for item in clusters[best_i]]
        means[best_i] = sum(xs) / len(xs)
    order = sorted(range(len(clusters)), key=lambda i: means[i])
    return [clusters[i] for i in order], [means[i] for i in order]


def _group_rows(words: list, y_tol: float = 4.5) -> list[list]:
    rows: list[list] = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        if rows and abs(word[1] - rows[-1][0][1]) <= y_tol:
            rows[-1].append(word)
        else:
            rows.append([word])
    return rows


def _assign_row(row_words: list, means: list[float]) -> list[str]:
    cells: list[list[str]] = [[] for _ in means]
    for word in row_words:
        x = (word[0] + word[2]) / 2
        index = min(range(len(means)), key=lambda i: abs(x - means[i]))
        cells[index].append(word[4])
    return [" ".join(parts) for parts in cells]


def _merge_wrapped_rows(rows: list[list[str]]) -> list[list[str]]:
    merged: list[list[str]] = []
    for cells in rows:
        leading = 0
        for cell in cells:
            if not cell.strip():
                leading += 1
            else:
                break
        only_last = leading >= max(1, len(cells) - 1)
        if merged and only_last:
            merged[-1] = [(prev + " " + cur).strip() for prev, cur in zip(merged[-1], cells)]
        else:
            merged.append(cells)
    return merged


def _matrix_to_rag_text(title: str, rows: list[list]) -> str:
    cleaned: list[list[str]] = []
    width = max(len(row) for row in rows)
    for row in rows:
        padded = list(row) + [""] * (width - len(row))
        cleaned.append([" ".join(str(cell or "").split()) for cell in padded[:width]])

    headers = cleaned[0]
    body = cleaned[1:]
    lines = [f"[Table] {title}".strip()]
    lines.append("Columns: " + " | ".join(headers))
    for row in body:
        pairs = [f"{head}: {cell}" for head, cell in zip(headers, row) if cell]
        if pairs:
            lines.append("- " + "; ".join(pairs))
    return "\n".join(lines)
