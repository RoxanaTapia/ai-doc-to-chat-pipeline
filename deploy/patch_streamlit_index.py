#!/usr/bin/env python3
"""Inject the first-paint boot loader into Streamlit's shipped index.html."""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "<!-- ai-doc-boot -->"


def _index_path() -> Path:
    import streamlit

    return Path(streamlit.__file__).resolve().parent / "static" / "index.html"


def patch(index: Path, css: str, js: str) -> bool:
    """Insert boot markup once. Returns True when the file changed."""
    text = index.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if "</head>" not in text or "<body>" not in text:
        raise SystemExit(f"unexpected Streamlit index.html shape: {index}")

    head = f"{MARKER}\n<style>\n{css.rstrip()}\n</style>\n"
    body = (
        f"{MARKER}\n"
        f'<div id="app-boot" role="status" aria-live="polite">'
        f'<span class="app-boot__label">Loading the document…</span>'
        f'<span class="app-boot__busy" aria-hidden="true">'
        f"<span></span><span></span><span></span>"
        f"</span>"
        f"</div>\n"
        f"<script>\n{js.rstrip()}\n</script>\n"
    )
    text = text.replace("</head>", f"{head}</head>", 1)
    text = text.replace("<body>", f"<body>\n{body}", 1)
    index.write_text(text, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    here = Path(__file__).resolve().parent
    css = (here / "streamlit_boot.css").read_text(encoding="utf-8")
    js = (here / "streamlit_boot.js").read_text(encoding="utf-8")
    target = Path(args[0]) if args else _index_path()
    changed = patch(target, css, js)
    print(f"{'patched' if changed else 'already-patched'}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
