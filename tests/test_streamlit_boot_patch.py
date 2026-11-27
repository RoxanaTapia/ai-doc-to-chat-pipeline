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
    assert 'id="app-boot"' in text
    assert "Reading the contract" in text
    assert "app-boot__dots" in text
    assert css in text
    assert js in text
    assert text.index("</style>") < text.index("</head>")
    assert text.index("app-boot") < text.index("</body>")
    assert 'src="/app/demo-kit/theme-init.js"' in text
    assert 'href="/app/demo-kit/favicon.svg"' in text
    assert 'href="/app/demo-kit/favicon-32.png"' in text
    assert 'href="/app/demo-kit/apple-touch-icon.png"' in text
    assert "plus-jakarta-sans-latin.woff2" in text
    assert "bricolage-grotesque-latin.woff2" in text
    assert 'href="/app/demo-kit/demo-kit.css"' in text
    assert 'class="dk-header"' in text
    assert "Ask your document" in text
    assert "Portfolio demo · fictional business · sample data" in text
    assert "Back to portfolio" in text
    assert "Log out" in text
    assert 'id="demo-exit"' in text
    assert 'href="https://roxanatapia.dev/"' in text
    assert "/invite/exit" not in text
    assert "Exit demo" not in text
    before, after = text.split('id="demo-exit"', 1)
    exit_tag = before[before.rfind("<a") :] + 'id="demo-exit"' + after[: after.find(">") + 1]
    assert 'href="https://roxanatapia.dev/"' in exit_tag
    assert "target=" not in exit_tag
    assert "btn-primary" not in exit_tag
    assert 'src="/app/demo-kit/theme.js"' in text
    assert 'src="/app/demo-kit/motion.js"' in text
    assert "ops-heartbeat.js" not in text
    assert text.index("theme-init.js") < text.index("</style>")
    assert patch(index, css, js) is False


def test_boot_js_fail_safes_and_slow_copy() -> None:
    """Overlay must not cover exceptions forever; cold starts keep a longer loader."""
    js = (DEPLOY_DIR / "streamlit_boot.js").read_text(encoding="utf-8")
    assert 'BOOT_TEXT = "Reading the contract"' in js
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

    assert "app-boot__dots" in css
    assert "prefers-reduced-motion" in css
    assert "app-boot-dots" in css
    assert "animation: none" in css
    assert "#9d174d" in css
    assert "#f8f4f9" in css
    assert "#f7a8c6" in css
    assert "#14061f" in css
    assert "prefers-color-scheme: dark" in css
    assert 'class="app-boot__dots"' in patcher
    assert 'class="app-boot__label"' in patcher
    assert "theme-init.js" in patcher
    assert "install_server_hook" in patcher
    ops = (DEPLOY_DIR / "ai_doc_ops.py").read_text(encoding="utf-8")
    assert 'OPS_HEADER = "X-Ops-Ok"' in ops
    assert "ops-heartbeat.js" in ops
    assert 'data-operator="true"' in ops


def test_operator_header_adds_flag_and_heartbeat() -> None:
    from ai_doc_ops import transform_index

    raw = b'<!DOCTYPE html><html lang="en"><body><p>Hi</p></body></html>'
    visitor = transform_index(raw, operator=False)
    assert visitor == raw
    operator = transform_index(raw, operator=True).decode("utf-8")
    assert '<html data-operator="true"' in operator
    assert 'src="/app/demo-kit/ops-heartbeat.js"' in operator
    again = transform_index(operator.encode("utf-8"), operator=True).decode("utf-8")
    assert again.count("data-operator=") == 1
    assert again.count("ops-heartbeat.js") == 1


def _handler_bytes(content: object) -> bytes:
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    return b"".join(content)  # type: ignore[arg-type]


def test_ops_hook_rewrites_only_operator_index(tmp_path: Path) -> None:
    """The static handler keeps visitors unchanged and rewrites an operator index."""
    import tornado.web
    from ai_doc_ops import install

    page = tmp_path / "index.html"
    page.write_text("<html><body>Hi</body></html>", encoding="utf-8")
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"plain")

    class Handler(tornado.web.StaticFileHandler):
        pass

    install(Handler)
    hashed = _handler_bytes(Handler.get_content(str(page)))
    assert b"data-operator" not in hashed

    class _Request:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    def _fake(path: Path, headers: dict[str, str]) -> Handler:
        handler = Handler.__new__(Handler)
        handler.absolute_path = str(path)
        handler.request = _Request(headers)  # type: ignore[attr-defined]
        return handler

    visitor = _fake(page, {})
    operator = _fake(page, {"X-Ops-Ok": "1"})
    asset = _fake(plain, {"X-Ops-Ok": "1"})

    visitor_body = _handler_bytes(Handler.get_content(visitor, str(page)))
    operator_body = _handler_bytes(Handler.get_content(operator, str(page)))
    assert b"data-operator" not in visitor_body
    assert b"ops-heartbeat.js" not in visitor_body
    assert b'data-operator="true"' in operator_body
    assert b"/app/demo-kit/ops-heartbeat.js" in operator_body
    assert Handler.get_content_size(operator) == len(operator_body)
    assert Handler.get_content_size(visitor) == page.stat().st_size
    assert Handler.compute_etag(operator) != Handler.compute_etag(visitor)
    assert _handler_bytes(Handler.get_content(asset, str(plain))) == b"plain"


def test_vendored_demo_kit_matches_sums() -> None:
    import subprocess

    script = PROJECT_ROOT / "vendor" / "demo-kit" / "check-hash.sh"
    completed = subprocess.run(["sh", str(script)], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    version = (PROJECT_ROOT / "vendor" / "demo-kit" / "VERSION").read_text(encoding="utf-8")
    assert version == "2\n"
    kit_css = (PROJECT_ROOT / "vendor" / "demo-kit" / "demo-kit.css").read_text(encoding="utf-8")
    motion = (PROJECT_ROOT / "vendor" / "demo-kit" / "motion.js").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in kit_css
    assert "reducedMotion" in motion or "prefers-reduced-motion" in motion
    assert "fonts.googleapis.com" not in kit_css
