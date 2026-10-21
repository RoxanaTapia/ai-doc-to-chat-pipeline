"""Unreadable PDF uploads: friendly copy, drop bad file, Ask stays off."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from page_copy import _CLIENT_COPY  # noqa: E402
from rag.ingestion import extract_pdf  # noqa: E402

FRIENDLY = "We couldn't read this PDF. Try another file."


def test_pdf_unreadable_copy_is_plain() -> None:
    assert _CLIENT_COPY["pdf_unreadable"] == FRIENDLY
    assert "/tmp" not in FRIENDLY
    assert "Failed to open" not in FRIENDLY


def test_extract_pdf_rejects_corrupt_bytes(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nnot a real pdf body")
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf(path, enable_ocr=False)
    detail = str(exc_info.value)
    assert "Failed to open" in detail or "pdf" in detail.lower()
    assert str(path) in detail


def test_extract_pdf_rejects_renamed_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_text("this is a plain text file renamed to look like a pdf\n")
    with pytest.raises(RuntimeError) as exc_info:
        extract_pdf(path, enable_ocr=False)
    detail = str(exc_info.value)
    assert str(path) in detail


def test_mark_upload_unreadable_drops_file_and_index() -> None:
    """After a bad upload, cache and index are gone so Ask cannot answer from the sample."""
    import session_upload
    from session_upload import mark_upload_unreadable

    class _State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def pop(self, key, default=None):
            return dict.pop(self, key, default)

    fake = _State(
        vector_store=object(),
        chunks=["leftover sample"],
        current_file="sample-nda.pdf",
        last_processed_name="sample-nda.pdf",
        last_processed_hash="abc",
        uploaded_pdf_bytes=b"%PDF-1.4\nbad",
        uploaded_pdf_name="corrupt.pdf",
        uploaded_pdf_hash="deadbeef",
        bm25_state=None,
        indexed_doc_stats={"file_name": "sample-nda.pdf", "pages": 3},
        last_generation_error=None,
        sample_preload_failed=False,
        uploader_key_version=2,
        own_pdf_mode=True,
        show_own_pdf=True,
        _uploader_widget_had_file=True,
        messages=[{"role": "user", "content": "hi"}],
        last_query="hi",
        last_retrieved_docs=[],
        last_retrieval_mode=None,
        last_raw_results=[],
        last_context="",
        last_answer=None,
        last_retrieval_metrics=None,
        last_response_timing=None,
        doc_status="ready",
        doc_status_detail="Sample NDA · 3 pages",
        developer_mode=False,
    )
    previous = session_upload.st.session_state
    session_upload.st.session_state = fake
    try:
        mark_upload_unreadable()
        assert fake.vector_store is None
        assert fake.chunks is None
        assert "uploaded_pdf_bytes" not in fake
        assert "uploaded_pdf_name" not in fake
        assert "uploaded_pdf_hash" not in fake
        assert fake.uploader_key_version == 3
        assert fake._uploader_widget_had_file is False
        assert fake.doc_status == "error"
        assert fake.doc_status_detail == FRIENDLY
        assert "/tmp" not in fake.doc_status_detail
        assert "Failed to open" not in fake.doc_status_detail
        assert fake.sample_preload_failed is True
        assert fake.own_pdf_mode is True
        assert fake.show_own_pdf is True
        assert fake.messages == []
    finally:
        session_upload.st.session_state = previous


def test_refusal_timing_is_skipped() -> None:
    """Refusal / not-covered cards must not show a misleading Answered in 0.01s."""
    import sources_panel
    from rag.citations import INSUFFICIENT_CONTEXT_ANSWER
    from sources_panel import _render_answer_card

    calls: list[object] = []

    class _FakeSt:
        def container(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def markdown(self, *args, **kwargs):
            return None

    previous_st = sources_panel.st
    previous_render = sources_panel._render_answer_timing
    sources_panel.st = _FakeSt()
    sources_panel._render_answer_timing = lambda *a, **k: calls.append((a, k))
    try:
        _render_answer_card(
            question="What is the penalty fee?",
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            sources=[],
            refusal=True,
            timing={"total_ms": 12.0, "retrieval_ms": 12.0, "generation_ms": 0.0},
            developer_mode=False,
            page_count=3,
            card_key="t",
        )
        assert calls == []
        _render_answer_card(
            question="How long?",
            answer="3 years",
            sources=None,
            refusal=False,
            timing={"total_ms": 1200.0, "retrieval_ms": 200.0, "generation_ms": 1000.0},
            developer_mode=False,
            page_count=3,
            card_key="t2",
        )
        assert len(calls) == 1
    finally:
        sources_panel.st = previous_st
        sources_panel._render_answer_timing = previous_render
