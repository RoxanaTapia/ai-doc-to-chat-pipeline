"""PDF ingest: page text, optional OCR, page numbers on every passage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from langchain_core.documents import Document

from ocr import is_likely_scanned_page, ocr_page_text


@dataclass(frozen=True)
class ExtractedPdf:
    page_docs: list[Document]
    preview_text: str
    scanned_pages_detected: int
    ocr_pages_attempted: int
    ocr_pages_used: int
    ocr_warning: str | None


def extract_pdf(path: Path, *, enable_ocr: bool) -> ExtractedPdf:
    """Read a PDF into one Document per page. Page numbers stay on metadata."""
    doc = fitz.open(str(path))
    page_docs: list[Document] = []
    preview_parts: list[str] = []
    scanned_pages_detected = 0
    ocr_pages_attempted = 0
    ocr_pages_used = 0
    ocr_warning: str | None = None

    try:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            final_page_text = page_text
            if enable_ocr and is_likely_scanned_page(page_text):
                scanned_pages_detected += 1
                ocr_pages_attempted += 1
                ocr_text, ocr_error = ocr_page_text(page)
                if ocr_error and ocr_warning is None:
                    ocr_warning = ocr_error
                if ocr_text:
                    final_page_text = ocr_text
                    ocr_pages_used += 1
            preview_parts.append(f"\n--- Page {page_num} ---\n{final_page_text}\n")
            page_docs.append(
                Document(page_content=final_page_text, metadata={"page": page_num})
            )
    finally:
        doc.close()

    return ExtractedPdf(
        page_docs=page_docs,
        preview_text="".join(preview_parts),
        scanned_pages_detected=scanned_pages_detected,
        ocr_pages_attempted=ocr_pages_attempted,
        ocr_pages_used=ocr_pages_used,
        ocr_warning=ocr_warning,
    )
