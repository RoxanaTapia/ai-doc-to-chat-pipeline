"""Answer-first demo helpers: copy, preload, headline, highlight, booking."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document  # noqa: E402

from page_copy import (  # noqa: E402
    _CLIENT_COPY,
    BOOKING_URL_ENV,
    booking_cta,
    booking_url,
    checked_all_pages_note,
    privacy_badge_for_provider,
    privacy_line_for_provider,
    sample_preload_error_copy,
    should_preload_sample,
    should_show_booking,
    should_show_starter_chips,
)
from page_preview import highlight_rects, page_index, render_highlighted_page  # noqa: E402
from rag.citations import INSUFFICIENT_CONTEXT_ANSWER, is_insufficient_answer  # noqa: E402
from session_upload import page_text_counts  # noqa: E402
from sources_panel import (  # noqa: E402
    display_answer_parts,
    split_answer_headline,
    strip_headline_markup,
)
from ui_theme import _HIDE_SIDEBAR_CSS, _THEME_CSS  # noqa: E402

SECTION_4_SENTENCE = (
    "This Agreement shall remain in effect for a period of three (3) years "
    "from the date of signing."
)
GOVERNING_LAW_QUOTE = (
    "This Agreement shall be governed by the laws of Spain. "
    "Any dispute arising hereunder shall be resolved exclusively in the courts of Madrid."
)

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
    assert privacy_badge_for_provider("ollama") == "Runs on this server"
    assert "offline" not in privacy_badge_for_provider("ollama").lower()


def test_refusal_sentinel_is_unchanged() -> None:
    assert INSUFFICIENT_CONTEXT_ANSWER == (
        "This isn't in the document, so I won't guess. Ask about something it covers."
    )
    assert is_insufficient_answer(INSUFFICIENT_CONTEXT_ANSWER)
    assert not is_insufficient_answer("3 years")


def test_checked_all_pages_note() -> None:
    assert checked_all_pages_note(3) == "Searched all 3 pages"
    assert checked_all_pages_note(1) == "Searched the whole page"
    assert checked_all_pages_note(0) == "Searched the document"
    assert checked_all_pages_note(3, any_blank=True) == "Searched the document"
    assert checked_all_pages_note(1, any_blank=True) == "Searched the document"


def test_should_preload_sample_only_on_first_paint() -> None:
    assert should_preload_sample(own_pdf_mode=False, has_upload=False, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=True, has_upload=False, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=False, has_upload=True, is_indexed=False)
    assert not should_preload_sample(own_pdf_mode=False, has_upload=False, is_indexed=True)


def test_booking_cta_hides_when_blank(monkeypatch) -> None:
    monkeypatch.delenv(BOOKING_URL_ENV, raising=False)
    empty = booking_cta("")
    assert empty["href"] is None
    assert empty["prompt"] == ""
    assert empty["action"] == ""
    assert not should_show_booking("")
    assert not should_show_booking("  ")
    assert booking_url("") == ""
    assert booking_url("  ") == ""

    filled = booking_cta("https://www.upwork.com/freelancers/roxanadev")
    assert filled["href"] == "https://www.upwork.com/freelancers/roxanadev"
    assert filled["prompt"] == "Want this on your documents?"
    assert filled["action"] == "Message me on Upwork."
    assert should_show_booking("https://www.upwork.com/freelancers/roxanadev")
    monkeypatch.setenv(BOOKING_URL_ENV, "https://www.upwork.com/freelancers/roxanadev")
    assert booking_url() == "https://www.upwork.com/freelancers/roxanadev"


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
    assert "max-width: 860px" in _THEME_CSS
    assert "0.72rem" not in _THEME_CSS
    assert "0.78rem" not in _THEME_CSS
    assert "0.82rem" not in _THEME_CSS
    assert 'section.main .block-container > div[data-testid="stVerticalBlock"]' in _THEME_CSS
    assert '[data-testid="column"]:has(.app-own-pdf) button' in _THEME_CSS
    assert 'div[data-testid="stVerticalBlock"]:has(.app-own-pdf) button' not in _THEME_CSS
    assert 'div[data-testid="stVerticalBlock"]:has(.app-answer-q)' not in _THEME_CSS
    assert 'div[data-testid="stVerticalBlock"]:has(.app-answer-card--refusal)' not in _THEME_CSS
    assert '[class*="st-key-answer_card_"]' in _THEME_CSS
    assert '[class*="st-key-refusal_card_"]' in _THEME_CSS
    assert '[class*="st-key-refusal_card_"] .app-answer-headline' in _THEME_CSS
    assert 'div[data-testid="stForm"] button' in _THEME_CSS
    assert '[class*="st-key-chip_"] button' in _THEME_CSS
    assert "max-width: 28rem" not in _THEME_CSS
    assert ".app-lock__line" in _THEME_CSS
    assert "max-width: none" in _THEME_CSS
    text = (SRC_DIR / "sources_panel.py").read_text(encoding="utf-8")
    assert 'st.container(key=f"{card_kind}_{card_key}")' in text
    assert 'card_kind = "refusal_card" if is_refusal else "answer_card"' in text


def test_readme_hosted_wording_does_not_overclaim() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    hosted = (
        "Your PDF and search stay on this server; only the few passages "
        "needed for an answer are sent to a hosted model (Anthropic). "
        "Files are deleted right after processing."
    )
    assert hosted in text
    assert text.startswith("# PDF Q&A with page citations\n")
    assert "On your own server, it can run fully offline with a local model." in text
    assert "on infrastructure you control" not in text
    assert "processed in memory and never stored" not in text
    assert "# Private PDF Q&A" not in text
    heading = text.split("\n", 1)[0].lower()
    assert "private" not in heading
    pilot = text.split("## Try the pilot", 1)[1].split("## ", 1)[0]
    hosted_block = pilot.split("On your own server", 1)[0].lower()
    assert "private" not in hosted_block
    assert "local" not in hosted_block
    arch = text.split("## How an answer is grounded", 1)[1].split("## ", 1)[0]
    assert "docs/assets/aidoc_architecture.png" in arch
    assert "caddy" not in arch.lower()
    assert "```mermaid" not in arch
    assert (PROJECT_ROOT / "docs" / "assets" / "aidoc_architecture.png").is_file()


def test_streamlit_keeps_base_path_and_no_usage_stats() -> None:
    text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'baseUrlPath = "app"' in text
    assert "gatherUsageStats = false" in text


def test_booking_env_is_documented() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DEMO_BOOKING_URL=https://www.upwork.com/freelancers/roxanadev" in example
    assert "cal.com" not in example.lower()
    assert "calendar" not in example.lower()


def test_starter_chips_hide_after_first_answer() -> None:
    assert should_show_starter_chips([])
    assert should_show_starter_chips([{"role": "user", "content": "How long?"}])
    assert not should_show_starter_chips(
        [
            {"role": "user", "content": "How long?"},
            {"role": "assistant", "content": "3 years"},
        ]
    )


def test_sample_preload_error_offers_retry() -> None:
    copy = sample_preload_error_copy()
    assert copy["error"] == "Could not load the sample NDA."
    assert copy["retry"] == "Use sample NDA"

    import session_upload
    from session_upload import mark_sample_preload_failed, render_sample_preload_error

    class _State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    fake = _State()
    previous = session_upload.st.session_state
    session_upload.st.session_state = fake
    try:
        mark_sample_preload_failed()
        assert fake.sample_preload_failed is True
        assert fake.doc_status == "error"
        assert "sample NDA" in fake.doc_status_detail
        assert callable(render_sample_preload_error)
    finally:
        session_upload.st.session_state = previous


def test_page_text_counts_ignore_blank_pages() -> None:
    docs = [
        Document(page_content="Section 4 - Term", metadata={"page": 1}),
        Document(page_content="   ", metadata={"page": 2}),
        Document(page_content="Spain / Madrid", metadata={"page": 3}),
    ]
    text_pages, blank_pages = page_text_counts(docs)
    assert text_pages == 2
    assert blank_pages == 1


def test_headline_strips_markdown_emphasis() -> None:
    assert strip_headline_markup("**3 years**") == "3 years"
    headline, body = display_answer_parts("**3 years**\n\nThe term runs from signing.")
    assert headline == "3 years"
    assert "signing" in body
    fallback, plain = display_answer_parts("**12 months**", "three (3) years from the date")
    assert fallback == ""
    assert "12 months" in plain


def test_page_index_is_one_based() -> None:
    assert page_index(2) == 1
    assert page_index("3") == 2
    assert page_index(0) is None
    assert page_index("x") is None


def test_working_timer_stays_in_developer_mode() -> None:
    text = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    assert "Working…" in text
    assert "script_ctx is not None and st.session_state.developer_mode" in text


def test_show_page_button_label_is_in_the_sources_panel() -> None:
    text = (SRC_DIR / "sources_panel.py").read_text(encoding="utf-8")
    assert '"Show page"' in text
    assert "click the quote" not in text.lower()


def test_highlight_covers_section_4_sentence() -> None:
    import pymupdf

    pdf_bytes = SAMPLE_NDA.read_bytes()
    png = render_highlighted_page(pdf_bytes, 2, SECTION_4_SENTENCE)
    assert png
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[1]
        expected = page.search_for(SECTION_4_SENTENCE)
        rects = highlight_rects(page, SECTION_4_SENTENCE)
    finally:
        doc.close()
    assert expected
    assert rects
    expected_area = sum(rect.get_area() for rect in expected)
    covered = 0.0
    for target in expected:
        for hit in rects:
            covered += (target & hit).get_area()
    assert covered + 1e-6 >= expected_area


def test_sample_nda_copy_matches_pdf() -> None:
    import pymupdf

    doc = pymupdf.open(SAMPLE_NDA)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    assert len(pages) == 3
    assert "Section 4 - Term" in pages[1]
    assert SECTION_4_SENTENCE in " ".join(pages[1].split())
    assert "Section 7 - Governing Law" in pages[2]
    assert GOVERNING_LAW_QUOTE in " ".join(pages[2].split())


PAGE_BG = (247, 247, 245)  # --app-bg / --app-paper
BEIGE_BG = (251, 246, 238)  # --app-warn-bg
INK = (26, 31, 36)  # --app-ink
BROWN = (107, 83, 48)  # --app-warn
SCREENSHOTS = PROJECT_ROOT / "docs" / "screenshots"


def _dist(pixel: tuple[int, ...], target: tuple[int, int, int]) -> int:
    return sum((int(pixel[i]) - target[i]) ** 2 for i in range(3))


def _closer_to(pixel: tuple[int, ...], a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return _dist(pixel, a) < _dist(pixel, b)


def test_storyboard_stills_keep_page_grey_and_ink_headlines() -> None:
    from PIL import Image

    page = Image.open(SCREENSHOTS / "05-page-and-privacy.png").convert("RGB")
    # Title sits on the page, not inside a refusal panel. Sample the band
    # behind "Ask your document" (upper third, content column).
    width, height = page.size
    samples = []
    for y in range(int(height * 0.06), int(height * 0.16), 4):
        for x in range(int(width * 0.08), int(width * 0.22), 6):
            pixel = page.getpixel((x, y))
            if pixel[0] + pixel[1] + pixel[2] > 680:
                samples.append(pixel)
    assert samples, "expected light page pixels behind the title"
    beige_hits = sum(1 for pixel in samples if _closer_to(pixel, BEIGE_BG, PAGE_BG))
    assert beige_hits * 4 < len(samples), "still 05 page background leaked beige"

    term = Image.open(SCREENSHOTS / "02-term-answer.png").convert("RGB")
    tw, th = term.size
    dark = []
    for y in range(int(th * 0.16), int(th * 0.38), 2):
        for x in range(int(tw * 0.06), int(tw * 0.28), 2):
            pixel = term.getpixel((x, y))
            if pixel[0] + pixel[1] + pixel[2] < 180:
                dark.append(pixel)
    assert dark, "expected headline pixels in still 02"
    brown_hits = sum(1 for pixel in dark if _closer_to(pixel, BROWN, INK))
    assert brown_hits * 4 < len(dark), "still 02 answer headline rendered brown"
