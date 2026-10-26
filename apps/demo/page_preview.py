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


def highlight_rects(page: pymupdf.Page, quote: str) -> list[pymupdf.Rect]:
    """Find quote rectangles on a page; empty when the passage is not visible."""
    for needle in _search_needles(quote):
        hits = page.search_for(needle, quads=False)
        if hits:
            return list(hits)
    return []


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
