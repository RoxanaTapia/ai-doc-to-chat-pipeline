"""Unit tests for the Streamlit index.html first-paint boot patch."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from patch_streamlit_index import MARKER, patch  # noqa: E402


def test_patch_injects_boot_once(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    index.write_text(
        "<!DOCTYPE html><html><head><title>t</title></head><body></body></html>",
        encoding="utf-8",
    )
    css = "[data-testid='stSkeleton'] { opacity: 0 !important; }"
    js = "document.getElementById('app-boot');"
    assert patch(index, css, js) is True
    text = index.read_text(encoding="utf-8")
    assert MARKER in text
    assert "id=\"app-boot\"" in text
    assert "Loading the document…" in text
    assert css in text
    assert 'src="./ai-doc-boot.js"' in text
    assert (tmp_path / "ai-doc-boot.js").read_text(encoding="utf-8").startswith(
        "document.getElementById('app-boot');"
    )
    assert text.index("</style>") < text.index("</head>")
    assert text.index("app-boot") < text.index("</body>")
    assert patch(index, css, js) is False


def test_boot_js_fail_safes_and_slow_copy() -> None:
    """Overlay must not cover exceptions forever; cold starts keep a longer loader."""
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    assert 'BOOT_TEXT = "Loading the document…"' in js
    assert 'SLOW_TEXT = "Still starting up…"' in js
    assert "SLOW_MS = 8000" in js
    assert "FAILSAFE_MS = 30000" in js
    assert 'data-testid="stException"' in js
    assert 'data-testid="stAlert"' in js
    assert "hasErrorSurface" in js
    assert "elapsed >= FAILSAFE_MS" in js
    assert "elapsed >= SLOW_MS" in js
    assert "connection error" in js
    # Ready path still requires the ask shell; fail-safes are extra dismisses.
    assert "stTextInput" in js
    assert "dismiss()" in js


def test_boot_js_dismiss_gate_waits_for_theme_fonts_and_frame() -> None:
    """Shell ready is not enough: hold until theme CSS applies, fonts settle, one frame."""
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    css = (DEPLOY_DIR / "streamlit_boot.css").read_text(encoding="utf-8")
    patcher = (DEPLOY_DIR / "patch_streamlit_index.py").read_text(encoding="utf-8")

    assert "THEME_GATE_MS = 4000" in js
    assert "themeApplied" in js
    assert '"--app-bg"' in js
    assert "settleThemeThenDismiss" in js
    assert "document.fonts" in js
    assert "fonts.ready" in js
    assert "requestAnimationFrame(() => dismiss())" in js
    # Ready path goes through the settle gate; errors / failsafe still dismiss now.
    assert "settleThemeThenDismiss()" in js
    ready_block = js.split("if (ready())", 1)[1].split("window.requestAnimationFrame(tick)", 1)[0]
    assert "settleThemeThenDismiss()" in ready_block
    assert "dismiss()" not in ready_block.replace("settleThemeThenDismiss()", "")

    assert "app-boot__busy" in css
    assert "prefers-reduced-motion" in css
    assert "app-boot-pulse" in css
    assert 'class="app-boot__busy"' in patcher
    assert 'class="app-boot__label"' in patcher


def test_boot_hides_prior_cards_on_ask_click() -> None:
    """Follow-up clicks hide old cards and show busy before the server round-trip."""
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    css = (DEPLOY_DIR / "streamlit_boot.css").read_text(encoding="utf-8")
    assert "app-ask-pending" in js
    assert "app-ask-pending" in css
    assert "st-key-ask_submit" in js
    assert "st-key-chip_" in js
    assert "Looking through the document" in css
    assert "app-ask-done" in js
    assert "__APP_ASK_TEST_HOOKS" in js
    assert "__APP_ASK_FAILSAFE_MS" in js
    assert "setTimeout" in js
    assert "restoreFully" in js
    assert "isComposing" in js
    assert "event.button" in js
    assert "trimmedAskValue" in js
    assert "st-key-answer_card_" in css
    assert "st-key-refusal_card_" in css
    assert "display: none !important" in css
    # Busy is a CSS ::before on .app-status (no orphan #app-ask-busy on <body>).
    assert "app-status::before" in css
    assert 'content: "Looking through the document"' in css
    assert 'getElementById("app-ask-busy")' not in js
    assert "id=\"app-ask-busy\"" not in js
    assert "#app-ask-busy" not in css
    # Must not rewrite Streamlit's status text/class.
    assert "status.textContent" not in js
    assert 'status.className = "app-status' not in js
    # Test hooks stay behind the flag.
    assert "if (testHooks())" in js or "testHooks()" in js
