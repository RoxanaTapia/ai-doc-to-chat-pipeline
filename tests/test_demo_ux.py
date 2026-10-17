"""Answer-first demo helpers: copy, preload, headline, highlight, booking."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from page_copy import (  # noqa: E402
    _CLIENT_COPY,
    BOOKING_URL_ENV,
    booking_cta,
    booking_url,
    checked_all_pages_note,
    privacy_badge_for_provider,
    privacy_line_for_provider,
    should_preload_sample,
)
from page_preview import highlight_rects, page_index, render_highlighted_page  # noqa: E402
from rag.citations import INSUFFICIENT_CONTEXT_ANSWER, is_insufficient_answer  # noqa: E402
from sources_panel import split_answer_headline  # noqa: E402
from ui_theme import _HIDE_SIDEBAR_CSS, _THEME_CSS  # noqa: E402

SAMPLE_NDA = PROJECT_ROOT / "docs" / "product" / "sample-nda.pdf"


def test_hero_is_one_subtitle() -> None:
    assert _CLIENT_COPY["hero_lead"] == (
        "Ask your contract anything. Every answer shows the page it came from."
    )
    assert not _CLIENT_COPY["hero_kicker"]


def test_privacy_wording_is_unchanged() -> None:
    assert privacy_line_for_provider("ollama") == (
        "Runs on this server: your PDF, search and answers never leave it. "
        "Files are deleted right after processing."
    )
    assert privacy_line_for_provider("anthropic") == (
        "Your PDF and search stay on this server; only the few passages "
        "needed for an answer are sent to a hosted model (Anthropic). "
        "Files are deleted right after processing."
    )
    assert privacy_line_for_provider("dummy") == "Files are deleted right after processing."


def test_anthropic_badge_never_says_private_local_or_offline() -> None:
    label = privacy_badge_for_provider("anthropic")
    assert label == "How your data is handled"
    lowered = label.lower()
    assert "private" not in lowered
    assert "local" not in lowered
    assert "offline" not in lowered
    assert privacy_badge_for_provider("dummy") == label
    assert privacy_badge_for_provider("openai") == label
    assert privacy_badge_for_provider("ollama") == "Runs offline"


def test_refusal_sentinel_is_unchanged() -> None:
    assert INSUFFICIENT_CONTEXT_ANSWER == (
        "This isn't in the document, so I won't guess. Ask about something it covers."
    )
    assert is_insufficient_answer(INSUFFICIENT_CONTEXT_ANSWER)
    assert not is_insufficient_answer("3 years")


def test_checked_all_pages_note() -> None:
    assert checked_all_pages_note(4) == "Checked all 4 pages"
    assert checked_all_pages_note(1) == "Checked the whole page"
    assert checked_all_pages_note(0) == "Checked the document"


def test_should_preload_sample_only_on_first_paint() -> None:
    assert should_preload_sample(own_pdf_mode=False, has_upload=False, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=True, has_upload=False, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=False, has_upload=True, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=False, has_upload=False, is_indexed=True)


def test_booking_cta_is_blank_safe(monkeypatch) -> None:
    monkeypatch.delenv(BOOKING_URL_ENV, raising=False)
    empty = booking_cta("")
    assert empty["prompt"] == "Want this on your documents?"
    assert empty["action"] == "Book a 15-min call."
    assert empty["href"] is None
    assert booking_url("") == ""
    assert booking_url("  ") == ""

    filled = booking_cta("https://cal.example/demo")
    assert filled["href"] == "https://cal.example/demo"
    monkeypatch.setenv(BOOKING_URL_ENV, "https://cal.example/from-env")
    assert booking_url() == "https://cal.example/from-env"


def test_split_answer_headline_prefers_short_first_line() -> None:
    headline, body = split_answer_headline("3 years\n\nThe term runs from the signing date.")
    assert headline == "3 years"
    assert "signing date" in body

    heading, rest = split_answer_headline("### Summary\nSpain / Madrid\nGoverning law is Spain.")
    assert heading == "Spain / Madrid"
    assert "Governing law" in rest

    only, empty = split_answer_headline(INSUFFICIENT_CONTEXT_ANSWER)
    assert only == INSUFFICIENT_CONTEXT_ANSWER
    assert empty == ""


def test_theme_hides_sidebar_and_keeps_one_accent() -> None:
    assert '[data-testid="stSidebar"]' in _HIDE_SIDEBAR_CSS
    assert "--app-accent: #3d4fd8" in _THEME_CSS
    assert "sans-serif" in _THEME_CSS
    assert "app-typing-dots" in _THEME_CSS


def test_streamlit_keeps_base_path_and_no_usage_stats() -> None:
    text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'baseUrlPath = "app"' in text
    assert "gatherUsageStats = false" in text


def test_booking_env_is_documented() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DEMO_BOOKING_URL=" in example


def test_page_index_is_one_based() -> None:
    assert page_index(2) == 1
    assert page_index("3") == 2
    assert page_index(0) is None
    assert page_index("x") is None


def test_highlight_finds_term_on_sample_nda() -> None:
    import pymupdf

    pdf_bytes = SAMPLE_NDA.read_bytes()
    png = render_highlighted_page(pdf_bytes, 2, "three (3) years")
    assert png
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        rects = highlight_rects(doc[1], "three (3) years")
    finally:
        doc.close()
    assert rects
