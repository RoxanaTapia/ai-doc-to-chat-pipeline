"""OCR helpers for scanned PDF pages.

Kept as a standalone module so it can be unit-tested without importing
the full Streamlit app. Only standard-library + optional OCR deps required.
"""
import io


def is_likely_scanned_page(text: str) -> bool:
    """Heuristic: very short extracted text suggests an image-only page."""
    return len((text or "").strip()) < 50


def ocr_page_text(page) -> tuple[str, str | None]:
    """Run Tesseract OCR on a PyMuPDF page.

    Args:
        page: A ``fitz.Page`` object.

    Returns:
        ``(ocr_text, error_message)`` — ``error_message`` is ``None`` on success.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", "OCR dependencies not installed (requires pytesseract + pillow + system tesseract)."

    try:
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        # Minimal preprocessing: simple thresholding helps many scanned contracts.
        image = image.point(lambda x: 0 if x < 180 else 255, mode="1")
        ocr_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6").strip()
        return ocr_text, None
    except (OSError, RuntimeError, ValueError) as exc:
        return "", f"OCR failed on page: {exc}"
