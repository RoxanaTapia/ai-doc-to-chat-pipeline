"""Screen-recording polish: chip preview, stable answers, booking, highlight, type."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
for _folder in (DEMO_DIR, SRC_DIR):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from page_copy import (  # noqa: E402
    BOOKING_URL_ENV,
    booking_cta,
    booking_url,
)
from page_preview import highlight_rects  # noqa: E402
from sources_panel import (  # noqa: E402
    cap_list_headline,
    headline_is_compact,
    quote_repeats_answer,
    source_chip_label,
)

SAMPLE_NDA = PROJECT_ROOT / "docs" / "product" / "sample-nda.pdf"
_EN_DASH = "\u2013"


def test_section_chip_uses_an_en_dash_and_sentence_case() -> None:
    label = source_chip_label(
        {
            "page": 2,
            "preview": "Section 4 - Term This Agreement shall remain in effect for three years.",
        }
    )
    assert label == f"Page 2 · Section 4 {_EN_DASH} Term"
    assert "\u2014" not in label
    panel = (DEMO_DIR / "sources_panel.py").read_text(encoding="utf-8")
    theme = (DEMO_DIR / "ui_theme.py").read_text(encoding="utf-8")
    preview_rule = theme.split("[class*=\"st-key-preview_\"]", 1)[1].split("}", 1)[0]
    assert "text-transform: uppercase" not in preview_rule
    assert "Answered in" in panel
    js = (DEMO_DIR / "ask_pending.js").read_text(encoding="utf-8")
    assert 'return "Answered in "' in js


def test_repeating_quote_is_dropped_and_a_different_excerpt_stays() -> None:
    answer = (
        "3 years\n\nThis Agreement shall remain in effect for a period of three (3) years "
        "from the date of signing."
    )
    padded = (
        "Section 4 - Term This Agreement shall remain in effect for a period of three (3) years "
        "from the date of signing. The obligations of confidentiality shall survive termination."
    )
    citation = (
        "Section 7 - Governing Law This Agreement shall be governed by the laws of Spain. "
        "Any dispute arising hereunder shall be resolved exclusively in the courts of Madrid."
    )
    law_answer = (
        "Spain / Madrid\n\nThis Agreement shall be governed by the laws of Spain. "
        "Any dispute arising hereunder shall be resolved exclusively in the courts of Madrid."
    )
    assert quote_repeats_answer(padded, answer)
    assert not quote_repeats_answer(citation, law_answer)
    assert not quote_repeats_answer("The filing fee is due on the first day.", answer)


def test_list_headline_stays_about_two_lines() -> None:
    long_line = (
        "- The receiving party must protect customer lists, financial projections, "
        "and marketing materials from anyone who does not need them for the work."
    )
    answer = long_line + "\n- Return or destroy the copies when the agreement ends."
    headline, body, compact = cap_list_headline(answer, long_line, "")
    assert compact
    assert len(headline) <= 72
    assert "customer lists" in headline or "customer lists" in body
    assert headline_is_compact(answer)
    short = "Four duties\n\n- Keep the files private.\n- Return them at the end."
    assert headline_is_compact(short) is False
    theme = (DEMO_DIR / "ui_theme.py").read_text(encoding="utf-8")
    assert "app-answer-headline--list" in theme
    assert "font-size: 1.25rem !important" in theme


def test_highlight_covers_the_sentence_past_customer_lists() -> None:
    """A preview cut at 'customer lists,' still highlights through the period."""
    import pymupdf

    from ai_doc.rag.citations import readable_excerpt

    doc = pymupdf.open(SAMPLE_NDA)
    try:
        page = doc[0]
        text = " ".join(page.get_text().split())
        chunk = text[text.lower().find("section 1") :]
        excerpt = readable_excerpt(chunk, 280)
        assert "customer lists," in excerpt.lower()
        assert "marketing materials" not in excerpt.lower()
        stopped = page.search_for("customer lists,")
        full = page.search_for("and marketing materials.")
        rects = highlight_rects(page, excerpt)
    finally:
        doc.close()
    assert stopped and rects and full
    covered = 0.0
    for target in full:
        for hit in rects:
            covered += (target & hit).get_area()
    full_area = sum(rect.get_area() for rect in full)
    assert covered + 1e-6 >= full_area * 0.9
    assert sum(rect.get_area() for rect in rects) > sum(rect.get_area() for rect in stopped)


def test_answering_keeps_the_document_line(monkeypatch: pytest.MonkeyPatch) -> None:
    import session_upload

    class _State(dict):
        def __getattr__(self, key: str):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    state = _State(
        vector_store=object(),
        current_file="sample-nda.pdf",
        indexed_doc_stats={"pages": 3},
    )
    monkeypatch.setattr(session_upload.st, "session_state", state)
    kind, message = session_upload._status_view("answering", "Looking through the document")
    assert kind == "ready"
    assert message.startswith("NDA · Acme Corp and Beta Labs")
    assert "Looking through" not in message
    assert not message.endswith(" .")


def test_loading_line_has_no_space_before_the_dot_and_chip_stays() -> None:
    theme = (DEMO_DIR / "ui_theme.py").read_text(encoding="utf-8")
    upload = (DEMO_DIR / "session_upload.py").read_text(encoding="utf-8")
    app = (DEMO_DIR / "app.py").read_text(encoding="utf-8")
    assert "gap: 0.15rem" not in theme
    assert 'content: var(--app-doc-answering) ".";' in theme
    assert "body.app-ask-pending .app-status" not in theme
    assert "class=\"app-ask-live\"" in app
    assert 'key="doc_chip"' in upload
    assert "Open page 1 of this document." in upload
    assert "sample_page_preview_open()" in app


def test_sticky_header_background_is_solid() -> None:
    theme = (DEMO_DIR / "ui_theme.py").read_text(encoding="utf-8")
    boot = (PROJECT_ROOT / "deploy" / "streamlit_boot.css").read_text(encoding="utf-8")
    assert ".dk-header" in theme
    assert "background: #f8f4f9 !important" in theme
    assert "background: #14061f !important" in theme
    assert "backdrop-filter: none !important" in theme
    assert "background: #f8f4f9 !important" in boot
    assert "background: #14061f !important" in boot
    kit = (PROJECT_ROOT / "vendor" / "demo-kit" / "demo-kit.css").read_text(encoding="utf-8")
    assert "rgba(248, 244, 249, 0.92)" in kit


def test_website_booking_is_a_new_tab_calendar_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BOOKING_URL_ENV, raising=False)
    cta = booking_cta()
    assert cta["action"] == "Book a call"
    assert cta["prompt"] == "Want this on your documents?"
    assert cta["href"].startswith("https://calendar.google.com/calendar/appointments/schedules/")
    assert cta["href"].endswith("?gv=true")
    upwork = booking_cta("https://www.upwork.com/freelancers/example")
    assert upwork["action"] == "Message me on Upwork."
    page = (DEMO_DIR / "page_copy.py").read_text(encoding="utf-8")
    footer = page.split("def client_footer_html", 1)[1]
    booking = page.split("def _render_booking_cta", 1)[1].split("def ", 1)[0]
    end = page.split("def _render_case_end", 1)[1].split("def ", 1)[0]
    assert 'target="_blank"' in booking
    assert "Book a call" in footer
    assert "Built by" in footer
    assert "booking_url()" in footer
    assert "booking_url()" in end
    assert "_PORTFOLIO_URL" not in end
    assert booking_url("") == ""


def test_sample_chip_opens_page_one(stub_env) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    labels = [str(getattr(btn, "label", "") or "") for btn in at.button]
    assert any(label.startswith("NDA · Acme Corp and Beta Labs") for label in labels)
    at.button("doc_chip").click().run(timeout=30)
    assert not at.exception
    assert at.session_state["open_page_preview"] == "doc-page-1"
    at.button("doc_chip").click().run(timeout=30)
    assert at.session_state["open_page_preview"] is None


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUB_EMBEDDINGS", "1")
    monkeypatch.setenv("DEMO_STUB_LLM", "1")
    monkeypatch.setenv("USE_DUMMY_GENERATOR", "1")
    monkeypatch.setenv("APP_PRESENTATION_MODE", "client")
    monkeypatch.delenv(BOOKING_URL_ENV, raising=False)
