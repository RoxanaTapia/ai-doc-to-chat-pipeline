"""Unit checks for #207 ask-pending delivery (st.html + theme, not the index patch)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ASK_JS = SRC_DIR / "ask_pending.js"
DEPLOY_DIR = PROJECT_ROOT / "deploy"


def test_ask_pending_does_not_extend_first_load_patch() -> None:
    """First-load patch stays as on main; ask-pending must not be wired into it."""
    boot_css = (DEPLOY_DIR / "streamlit_boot.css").read_text(encoding="utf-8")
    boot_js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    patcher = (DEPLOY_DIR / "patch_streamlit_index.py").read_text(encoding="utf-8")
    assert "app-ask-pending" not in boot_css
    assert "app-ask-pending" not in boot_js
    assert "ask_pending" not in patcher
    assert "ai-doc-boot.js" not in patcher
    assert "Looking through the document" not in boot_css


def test_ask_pending_js_guards_and_failsafe() -> None:
    js = ASK_JS.read_text(encoding="utf-8")
    # A raw '<' inside the payload starts an HTML tag and DOMPurify drops the script.
    assert "<" not in js
    assert "__appAskBootBound" in js
    assert "app-ask-pending" in js
    assert "app-ask-done" in js
    assert "st-key-ask_submit" in js
    assert "st-key-chip_" in js
    assert "__APP_ASK_TEST_HOOKS" in js
    assert "__APP_ASK_FAILSAFE_MS" in js
    assert "setTimeout" in js
    assert "restoreFully" in js
    assert "isComposing" in js
    assert "event.button" in js
    assert "trimmedAskValue" in js
    assert "status.textContent" not in js
    assert 'status.className = "app-status' not in js
    assert 'getElementById("app-ask-busy")' not in js


def test_theme_busy_uses_css_var_not_literal_copy() -> None:
    from page_copy import _CLIENT_COPY
    from ui_theme import _THEME_CSS, inject_theme

    assert "app-ask-pending" in _THEME_CSS
    assert "app-status::before" in _THEME_CSS
    assert "var(--app-doc-answering)" in _THEME_CSS
    assert 'content: "Looking through the document"' not in _THEME_CSS
    assert _CLIENT_COPY["doc_answering"] == "Looking through the document"
    src = (SRC_DIR / "ui_theme.py").read_text(encoding="utf-8")
    assert "unsafe_allow_javascript=True" in src
    assert "inject_ask_pending_client" in src
    assert "busy_label" in src
    assert inject_theme.__name__ == "inject_theme"
    page_copy = (SRC_DIR / "page_copy.py").read_text(encoding="utf-8")
    assert 'busy_label=_CLIENT_COPY["doc_answering"]' in page_copy


def test_app_timing_specificity_beats_markdown_p() -> None:
    from ui_theme import _THEME_CSS

    assert '[data-testid="stMarkdownContainer"] p.app-timing' in _THEME_CSS
    assert "font-size: 0.9rem !important" in _THEME_CSS
    assert "color: var(--app-muted) !important" in _THEME_CSS


def test_stock_streamlit_has_no_ask_pending_in_index() -> None:
    """Ask-pending must not require rewriting Streamlit's installed index."""
    index = Path(st.__file__).resolve().parent / "static" / "index.html"
    text = index.read_text(encoding="utf-8")
    assert "app-ask-pending" not in text
    assert "Looking through the document" not in text
