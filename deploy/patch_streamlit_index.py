#!/usr/bin/env python3
"""Inject the first-paint boot loader and the brand shell into Streamlit's index."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

MARKER = "<!-- ai-doc-boot -->"
END_MARKER = "<!-- /ai-doc-boot -->"
KIT_URL_PREFIX = "/app/demo-kit"
HOOK_MARKER = "# ai-doc-ops-hook"
_SHORTCUT_ICON = re.compile(
    r"[ \t]*<link rel=\"shortcut icon\" href=\"\./favicon\.png\"\s*/>\s*",
)


def _index_path() -> Path:
    import streamlit

    return Path(streamlit.__file__).resolve().parent / "static" / "index.html"


def _kit_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "demo-kit"


def _header_html(prefix: str) -> str:
    raw = (Path(__file__).resolve().parent / "brand_header.html").read_text(encoding="utf-8")
    return raw.replace("__KIT__", prefix).strip()


def _copy_kit(index: Path) -> None:
    """Serve the vendored kit next to Streamlit's index, same origin."""
    src = _kit_dir()
    if not src.is_dir():
        return
    dest = index.parent / "demo-kit"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _brand_early(prefix: str) -> str:
    # Blocking script: data-theme is set before the rest of <head> is fetched.
    return (
        f"{MARKER}\n"
        f'<script src="{prefix}/theme-init.js"></script>\n'
        f"{END_MARKER}\n"
    )


def _brand_head(prefix: str, css: str) -> str:
    return (
        f"{MARKER}\n"
        '<meta name="theme-color" content="#f8f4f9" />\n'
        f'<link rel="icon" href="{prefix}/favicon.svg" type="image/svg+xml" />\n'
        f'<link rel="icon" href="{prefix}/favicon-32.png" type="image/png" sizes="32x32" />\n'
        f'<link rel="apple-touch-icon" href="{prefix}/apple-touch-icon.png" />\n'
        f'<link rel="preload" href="{prefix}/fonts/plus-jakarta-sans-latin.woff2" '
        'as="font" type="font/woff2" />\n'
        f'<link rel="preload" href="{prefix}/fonts/bricolage-grotesque-latin.woff2" '
        'as="font" type="font/woff2" />\n'
        f'<link rel="stylesheet" href="{prefix}/demo-kit.css" />\n'
        f"<style>\n{css.rstrip()}\n</style>\n"
        f"{END_MARKER}\n"
    )


def _brand_body(prefix: str, js: str) -> str:
    header = _header_html(prefix)
    return (
        f"{MARKER}\n"
        f"{header}\n"
        f'<div id="app-boot" role="status" aria-live="polite">'
        f'<p class="app-boot__line">'
        f'<span class="app-boot__label">Reading the contract</span>'
        f'<span class="app-boot__dots" aria-hidden="true"></span>'
        f"</p>"
        f"</div>\n"
        f"<script>\n{js.rstrip()}\n</script>\n"
        f'<script src="{prefix}/theme.js"></script>\n'
        f'<script src="{prefix}/motion.js"></script>\n'
        f"{END_MARKER}\n"
    )


def patch(index: Path, css: str, js: str, *, prefix: str = KIT_URL_PREFIX) -> bool:
    """Insert boot and brand markup once. Returns True when the file changed."""
    _copy_kit(index)
    text = index.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if "</head>" not in text or "<body>" not in text or "<head>" not in text:
        raise SystemExit(f"unexpected Streamlit index.html shape: {index}")

    text = _SHORTCUT_ICON.sub("", text)
    text = text.replace("<head>", "<head>\n" + _brand_early(prefix), 1)
    text = text.replace("</head>", _brand_head(prefix, css) + "</head>", 1)
    text = text.replace("<body>", "<body>\n" + _brand_body(prefix, js), 1)
    index.write_text(text, encoding="utf-8")
    return True


def _server_dir() -> Path:
    import streamlit

    return Path(streamlit.__file__).resolve().parent / "web" / "server"


def install_server_hook() -> bool:
    """Copy the operator rewriter beside Streamlit and call it from routes.py."""
    server = _server_dir()
    source = Path(__file__).resolve().parent / "ai_doc_ops.py"
    target = server / "ai_doc_ops.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    routes = server / "routes.py"
    text = routes.read_text(encoding="utf-8")
    if HOOK_MARKER in text:
        return False
    snippet = (
        f"\n{HOOK_MARKER}\n"
        "from streamlit.web.server.ai_doc_ops import install as _ai_doc_install_ops\n"
        "\n"
        "_ai_doc_install_ops(StaticFileHandler)\n"
    )
    routes.write_text(text + snippet, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    here = Path(__file__).resolve().parent
    css = (here / "streamlit_boot.css").read_text(encoding="utf-8")
    js = (here / "streamlit_boot.js").read_text(encoding="utf-8")
    target = Path(args[0]) if args else _index_path()
    changed = patch(target, css, js)
    hooked = install_server_hook()
    print(f"{'patched' if changed else 'already-patched'}: {target}")
    print(f"{'hooked' if hooked else 'already-hooked'}: operator index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
