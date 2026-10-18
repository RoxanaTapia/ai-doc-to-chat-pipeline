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


def test_privacy_wording_matches_session_lifetime() -> None:
    """Pin the visitor lines so they cannot claim a delete the code does not do."""
    ollama = privacy_line_for_provider("ollama")
    anthropic = privacy_line_for_provider("anthropic")
    dummy = privacy_line_for_provider("dummy")
    assert ollama == (
        "Runs on this server: your PDF, search and answers never leave it. "
        "Kept only for this browser session, and cleared when you leave or reset."
    )
    assert anthropic == (
        "Your PDF and search stay on this server for this browser session; "
        "only the few passages needed for an answer are sent to a hosted model "
        "(Anthropic). Cleared when you leave or reset."
    )
    assert dummy == (
        "Your PDF is kept only for this browser session, and cleared when you leave or reset."
    )
    for line in (ollama, anthropic, dummy):
        assert "deleted" not in line.lower()
        assert "dsgvo" not in line.lower()
        assert "gdpr" not in line.lower()
    assert "never leave" not in anthropic
    assert "passages" not in ollama
    assert "Anthropic" not in ollama


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
    assert '[class*="st-key-answer_card_"] .app-answer-headline' in _THEME_CSS
    assert '[class*="st-key-refusal_card_"] .app-answer-headline' in _THEME_CSS
    assert "font-size: 2rem !important" in _THEME_CSS
    assert "font-size: 1.2rem !important" in _THEME_CSS
    assert '[class*="st-key-ask_submit"] button' in _THEME_CSS
    assert '[class*="st-key-chip_"] button' in _THEME_CSS
    assert '[data-testid="InputInstructions"]' in _THEME_CSS
    assert (
        'div[data-testid="stVerticalBlock"]:has(.app-ask-mark) '
        '[data-testid="InputInstructions"]'
    ) in _THEME_CSS
    assert "max-width: 28rem" not in _THEME_CSS
    assert ".app-lock__line" in _THEME_CSS
    assert ".app-toolbar-mark" in _THEME_CSS
    assert "--app-header-min-height" in _THEME_CSS
    assert "--app-status-line: 1.5rem" in _THEME_CSS
    assert "opening the note must not grow the header row" in _THEME_CSS
    assert "flex-wrap: nowrap !important" in _THEME_CSS
    assert "@media (max-width: 40rem)" in _THEME_CSS
    assert ".app-lock__line {\n  position: absolute;" in _THEME_CSS
    assert "position: static !important" in _THEME_CSS
    assert "text-overflow: ellipsis" not in _THEME_CSS
    assert ".app-status {\n  display: block;" in _THEME_CSS
    assert "white-space: normal" in _THEME_CSS
    assert "min-height: var(--app-status-line)" in _THEME_CSS
    assert "overflow: hidden !important" not in _THEME_CSS
    assert "overflow: visible;" in _THEME_CSS
    desktop, _, mobile = _THEME_CSS.partition("@media (max-width: 40rem)")
    assert "flex: 1 1 100%" not in desktop
    assert ".app-lock[open] {\n    display: block;\n    width: 100%;\n  }" in mobile
    assert "position: static;" in mobile
    assert "flex: 1 1 100% !important" in mobile
    assert "never an overlay that covers status / own-PDF" in mobile
    text = (SRC_DIR / "sources_panel.py").read_text(encoding="utf-8")
    assert 'st.container(key=f"{card_kind}_{card_key}")' in text
    assert 'card_kind = "refusal_card" if is_refusal else "answer_card"' in text


def test_readme_hosted_wording_does_not_overclaim() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    hosted = (
        "Your PDF and search stay on this server for this browser session; "
        "only the few passages needed for an answer are sent to a hosted model "
        "(Anthropic). Cleared when you leave or reset."
    )
    assert hosted in text
    assert text.startswith("# PDF Q&A with page citations\n")
    assert "On your own server, it can run fully offline with a local model." in text
    assert "on infrastructure you control" not in text
    assert "processed in memory and never stored" not in text
    assert "# Private PDF Q&A" not in text
    assert "**On-server embeddings**" in text
    assert "**Private embeddings**" not in text
    heading = text.split("\n", 1)[0].lower()
    assert "private" not in heading
    script = (PROJECT_ROOT / "docs" / "product" / "demo-script.md").read_text(encoding="utf-8")
    assert "deleted right after processing" not in script
    assert "tap a starter question" in script
    pilot = text.split("## Try the pilot", 1)[1].split("## ", 1)[0]
    hosted_block = pilot.split("On your own server", 1)[0].lower()
    assert "private" not in hosted_block
    assert "local" not in hosted_block
    assert "deleted" not in hosted_block
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
    assert should_show_starter_chips([], chat_ready=True)
    assert should_show_starter_chips([{"role": "user", "content": "How long?"}])
    assert not should_show_starter_chips([], chat_ready=False)
    assert not should_show_starter_chips(
        [{"role": "user", "content": "How long?"}],
        chat_ready=False,
    )
    assert not should_show_starter_chips(
        [
            {"role": "user", "content": "How long?"},
            {"role": "assistant", "content": "3 years"},
        ]
    )
    assert not should_show_starter_chips([], query="How long does this agreement last?")
    assert not should_show_starter_chips(
        [{"role": "user", "content": "How long?"}],
        query="How long?",
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


def test_ask_path_queues_then_generates_without_write_stream() -> None:
    text = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    assert "queued_question" in text
    assert "st.write_stream(" not in text
    assert "AskTimeoutError" in text
    assert text.find("run_with_timeout") < text.find("typing_slot.empty()")
    assert text.find("_collect_generated_answer") < text.find("typing_slot.empty()")
    assert 'paint_status(status_slot, "answering"' in text
    assert "Looking through the document" in _CLIENT_COPY["doc_answering"]
    assert _CLIENT_COPY["doc_timeout"] == ("That took too long, so I stopped. Please ask again.")


def test_header_row_keeps_privacy_status_and_own_pdf() -> None:
    """The three header cells stay siblings; the data note overlays only above 40rem."""
    text = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    assert 'class="app-toolbar-mark"' in text
    assert "st.columns([1.1, 2.2, 1.4])" in text
    assert "_render_privacy_badge(" in text
    assert "status_slot = status_col.empty()" in text
    assert 'key="own_pdf_toggle"' in text
    assert text.find('class="app-toolbar-mark"') < text.find("st.columns([1.1, 2.2, 1.4])")
    assert text.find("_render_privacy_badge") < text.find("status_slot = status_col.empty()")
    assert text.find("status_slot = status_col.empty()") < text.find('key="own_pdf_toggle"')
    badge = (SRC_DIR / "page_copy.py").read_text(encoding="utf-8")
    assert 'class="app-lock"' in badge
    assert 'class="app-lock__line"' in badge
    assert "<details" in badge
    assert "[data-testid=\"stHorizontalBlock\"]:has(.app-lock) .app-lock" in _THEME_CSS
    assert "right: 0;" in _THEME_CSS
    assert "max-width: 36rem" in _THEME_CSS


def test_ask_ui_has_no_form_that_can_flash_missing_submit() -> None:
    """Streamlit 1.54 paints 'Missing Submit Button' while a form hydrates."""
    text = (SRC_DIR / "app.py").read_text(encoding="utf-8")
    assert "st.form(" not in text
    assert "form_submit_button" not in text
    assert 'key="ask_submit"' in text
    assert "pending_question" not in text
    assert "queued_question = chip" not in text
    assert '_queue_question(chip["question"])' in text
    assert "def _render_starter_chips" in text
    assert "slot = st.empty()" in text
    assert "with slot.container()" in text
    assert "should_show_starter_chips(" in text
    assert "chat_ready=chat_ready" in text
    assert "chip_slot = _render_starter_chips(chat_ready=chat_ready, query=query)" in text
    assert "chip_slot.empty()" in text
    assert text.find("chip_slot = _render_starter_chips") < text.find("chip_slot.empty()")
    assert text.find("chip_slot.empty()") < text.find('paint_status(status_slot, "answering"')
    assert "Press Enter to apply" not in text
    assert "on_change=_on_ask_box_commit if chat_ready else None" in text
    assert "_continue_queued_ask(queued_question)" in text
    submit_at = text.find('key="ask_submit"')
    continue_at = text.find("_continue_queued_ask(queued_question)")
    assert submit_at != -1 and continue_at != -1 and submit_at < continue_at


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

    answer_h = _headline_ink_height(term, top_frac=0.12, bottom_frac=0.48)
    refusal = Image.open(SCREENSHOTS / "03-not-in-document.png").convert("RGB")
    refusal_h = _headline_ink_height(refusal, top_frac=0.18, bottom_frac=0.72)
    assert answer_h > refusal_h, (
        f"answer headline should be larger than refusal ({answer_h}px vs {refusal_h}px)"
    )


def _headline_ink_height(
    image,
    *,
    top_frac: float,
    bottom_frac: float,
    luma_max: int = 200,
) -> int:
    """Tallest dark-ink run in the headline band of a card still."""
    width, height = image.size
    ys: list[int] = []
    for y in range(int(height * top_frac), int(height * bottom_frac)):
        for x in range(int(width * 0.05), int(width * 0.72)):
            pixel = image.getpixel((x, y))
            if pixel[0] + pixel[1] + pixel[2] < luma_max:
                ys.append(y)
                break
    if not ys:
        return 0
    return max(ys) - min(ys) + 1
