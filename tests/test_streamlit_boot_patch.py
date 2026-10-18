"""Unit tests for the Streamlit index.html first-paint boot patch."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "deploy") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "deploy"))

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
