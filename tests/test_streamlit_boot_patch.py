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
    assert js in text
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
