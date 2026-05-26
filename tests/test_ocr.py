"""Unit tests for src/ocr.py.

No Streamlit, no Tesseract binary, no real PDF required.
All external OCR dependencies are mocked via sys.modules patching.

Run:
    pytest tests/test_ocr.py -v
"""
from pathlib import Path
import io
import sys
import types
import unittest.mock as mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr import is_likely_scanned_page, ocr_page_text  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pytesseract_stub(return_value: str = "extracted text") -> types.ModuleType:
    mod = types.ModuleType("pytesseract")
    mod.image_to_string = lambda *a, **kw: return_value  # type: ignore[attr-defined]
    return mod


class _FakePixmap:
    """Minimal fitz.Pixmap stand-in returning a 10×10 white PNG."""

    def tobytes(self, fmt: str) -> bytes:
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("L", (10, 10), 255).save(buf, format="PNG")
        return buf.getvalue()


class _FakePage:
    def get_pixmap(self, dpi: int) -> _FakePixmap:
        return _FakePixmap()


# ---------------------------------------------------------------------------
# is_likely_scanned_page
# ---------------------------------------------------------------------------

class TestIsLikelyScannedPage:
    def test_empty_string_is_scanned(self) -> None:
        assert is_likely_scanned_page("") is True

    def test_whitespace_only_is_scanned(self) -> None:
        assert is_likely_scanned_page("   \n\t  ") is True

    def test_none_treated_as_empty(self) -> None:
        assert is_likely_scanned_page(None) is True  # type: ignore[arg-type]

    def test_text_below_threshold_is_scanned(self) -> None:
        assert is_likely_scanned_page("a" * 49) is True

    def test_text_at_boundary_is_not_scanned(self) -> None:
        assert is_likely_scanned_page("a" * 50) is False

    def test_normal_paragraph_is_not_scanned(self) -> None:
        text = "This Agreement is entered into by and between Party A and Party B."
        assert is_likely_scanned_page(text) is False

    def test_leading_trailing_whitespace_ignored(self) -> None:
        # 10 real chars — well below threshold even without surrounding spaces
        assert is_likely_scanned_page("  " + "a" * 10 + "  ") is True


# ---------------------------------------------------------------------------
# ocr_page_text
# ---------------------------------------------------------------------------

class TestOcrPageText:
    def test_returns_stripped_text_and_no_error_on_success(self) -> None:
        stub = _make_pytesseract_stub("  Extracted text  ")
        with mock.patch.dict(sys.modules, {"pytesseract": stub}):
            text, err = ocr_page_text(_FakePage())
        assert err is None
        assert text == "Extracted text"

    def test_returns_empty_string_and_no_error_on_blank_page(self) -> None:
        stub = _make_pytesseract_stub("   ")
        with mock.patch.dict(sys.modules, {"pytesseract": stub}):
            text, err = ocr_page_text(_FakePage())
        assert err is None
        assert text == ""

    def test_returns_error_message_when_pytesseract_import_fails(self) -> None:
        with mock.patch.dict(sys.modules, {"pytesseract": None}):  # type: ignore[dict-item]
            text, err = ocr_page_text(_FakePage())
        assert text == ""
        assert err is not None
        assert "OCR dependencies not installed" in err

    def test_returns_error_message_on_runtime_error(self) -> None:
        def _raise(*a, **kw):
            raise RuntimeError("tesseract binary not found")

        stub = _make_pytesseract_stub()
        stub.image_to_string = _raise  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"pytesseract": stub}):
            text, err = ocr_page_text(_FakePage())
        assert text == ""
        assert err is not None
        assert "OCR failed on page" in err
        assert "tesseract binary not found" in err

    def test_returns_error_message_on_os_error(self) -> None:
        def _raise(*a, **kw):
            raise OSError("cannot open image")

        stub = _make_pytesseract_stub()
        stub.image_to_string = _raise  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"pytesseract": stub}):
            text, err = ocr_page_text(_FakePage())
        assert text == ""
        assert "OCR failed on page" in (err or "")

    def test_returns_error_message_on_value_error(self) -> None:
        def _raise(*a, **kw):
            raise ValueError("unsupported image mode")

        stub = _make_pytesseract_stub()
        stub.image_to_string = _raise  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"pytesseract": stub}):
            text, err = ocr_page_text(_FakePage())
        assert text == ""
        assert "OCR failed on page" in (err or "")
