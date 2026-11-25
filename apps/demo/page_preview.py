"""Render a PDF page with the cited passage highlighted."""

from __future__ import annotations

import re

import pymupdf

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def page_index(page_number: object) -> int | None:
    """Turn 1-based page metadata into a 0-based document index."""
    try:
        number = int(page_number)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number - 1


def _search_needles(quote: str) -> list[str]:
    """Longest-first snippets that PyMuPDF can look up on a page."""
    cleaned = " ".join((quote or "").split())
    if not cleaned:
        return []
    needles = [cleaned]
    if len(cleaned) > 80:
        needles.append(cleaned[:80].rsplit(" ", 1)[0] or cleaned[:80])
    words = _WORD_RE.findall(cleaned)
    if len(words) >= 4:
        needles.append(" ".join(words[:6]))
    if len(words) >= 2:
        needles.append(" ".join(words[:3]))
    seen: set[str] = set()
    ordered: list[str] = []
    for needle in needles:
        key = needle.lower()
        if key in seen or len(needle) < 3:
            continue
        seen.add(key)
        ordered.append(needle)
    return ordered


def _quote_words(quote: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(quote or "")]


def _page_words(page: pymupdf.Page) -> list[tuple]:
    """Word boxes as (x0, y0, x1, y1, word, block, line)."""
    rows = []
    for item in page.get_text("words") or []:
        tokens = _WORD_RE.findall(str(item[4]))
        if not tokens:
            continue
        rows.append(
            (
                float(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                tokens[0].lower(),
                int(item[5]),
                int(item[6]),
            )
        )
    return rows


def _matching_word_run(page_words: list[tuple], quote_words: list[str]) -> list[tuple]:
    """Earliest page run that contains the quote words in order."""
    if len(quote_words) < 4 or not page_words:
        return []
    limit = len(page_words)
    for start, row in enumerate(page_words):
        if row[4] != quote_words[0]:
            continue
        if start + len(quote_words) > limit:
            break
        if all(page_words[start + offset][4] == word for offset, word in enumerate(quote_words)):
            return page_words[start : start + len(quote_words)]
    return []


def _line_rects(matches: list[tuple]) -> list[pymupdf.Rect]:
    """One rectangle per line so the highlight covers the whole passage."""
    lines: dict[tuple[int, int], list[float]] = {}
    for x0, y0, x1, y1, _word, block, line in matches:
        box = lines.get((block, line))
        if box is None:
            lines[(block, line)] = [x0, y0, x1, y1]
            continue
        box[0] = min(box[0], x0)
        box[1] = min(box[1], y0)
        box[2] = max(box[2], x1)
        box[3] = max(box[3], y1)
    return [pymupdf.Rect(box) for box in lines.values()]


def highlight_rects(page: pymupdf.Page, quote: str) -> list[pymupdf.Rect]:
    """Find quote rectangles on a page; empty when the passage is not visible.

    A full-string hit wins. When the excerpt does not match as one string
    (a heading glued on, or a dash the page does not use), the word run
    still highlights every line of the passage instead of the first few words.
    """
    cleaned = " ".join((quote or "").split())
    if not cleaned:
        return []
    full = page.search_for(cleaned, quads=False)
    if full:
        return list(full)
    words = _quote_words(cleaned)
    run = _matching_word_run(_page_words(page), words)
    if run:
        return _line_rects(run)
    for needle in _search_needles(cleaned)[1:]:
        hits = page.search_for(needle, quads=False)
        if hits:
            return list(hits)
    return []


def highlight_top_ratio(pdf_bytes: bytes, page_number: object, quote: str) -> float | None:
    """Fraction down the page where the highlight starts, for scrolling it into view."""
    if not pdf_bytes:
        return None
    index = page_index(page_number)
    if index is None:
        return None
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if index >= doc.page_count:
            return None
        page = doc[index]
        rects = highlight_rects(page, quote)
        if not rects:
            return None
        height = float(page.rect.height) or 1.0
        top = min(float(rect.y0) for rect in rects)
        return max(0.0, min(0.95, top / height))
    finally:
        doc.close()


def render_highlighted_page(
    pdf_bytes: bytes,
    page_number: object,
    quote: str,
    *,
    scale: float = 1.6,
) -> bytes | None:
    """PNG of ``page_number`` with ``quote`` highlighted, or None if the page is missing."""
    if not pdf_bytes:
        return None
    index = page_index(page_number)
    if index is None:
        return None
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if index >= doc.page_count:
            return None
        page = doc[index]
        for rect in highlight_rects(page, quote):
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(1.0, 0.92, 0.45))
            annot.update()
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()
