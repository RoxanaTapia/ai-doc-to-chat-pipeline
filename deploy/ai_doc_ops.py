"""Mark an operator session on Streamlit's index before first paint.

The edge sends ``X-Ops-Ok`` on the document request. Static ``index.html``
cannot see that header, so the static-file handler rewrites the page:
``data-operator="true"`` and the kit heartbeat script. Visitors are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

OPS_HEADER = "X-Ops-Ok"
HEARTBEAT_SRC = "/app/demo-kit/ops-heartbeat.js"
_MISSING = object()


def transform_index(raw: bytes, *, operator: bool, script_src: str = HEARTBEAT_SRC) -> bytes:
    """Return index bytes, with the operator flag and heartbeat when requested."""
    if not operator:
        return raw
    text = raw.decode("utf-8")
    if 'data-operator="' not in text:
        text = text.replace("<html", '<html data-operator="true"', 1)
    if "ops-heartbeat.js" not in text:
        tag = f'<script src="{script_src}"></script>\n'
        text = text.replace("</body>", f"{tag}</body>", 1)
    return text.encode("utf-8")


def install(handler_cls: type) -> None:
    """Wrap a Tornado static handler so index.html follows ``X-Ops-Ok``."""
    if getattr(handler_cls, "_ai_doc_ops_installed", False):
        return

    orig_size = handler_cls.get_content_size
    orig_content = handler_cls.get_content
    orig_etag = getattr(handler_cls, "compute_etag", None)

    def _is_index(self: Any) -> bool:
        path = getattr(self, "absolute_path", "") or ""
        return path.endswith(f"{os.sep}index.html") or path.endswith("/index.html")

    def _operator(self: Any) -> bool:
        request = getattr(self, "request", None)
        headers = getattr(request, "headers", None)
        if headers is None:
            return False
        return headers.get(OPS_HEADER) is not None

    def _body(self: Any) -> bytes | None:
        cached = getattr(self, "_ai_doc_ops_body", _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        if not _is_index(self) or not _operator(self):
            self._ai_doc_ops_body = None
            return None
        raw = orig_content(self.absolute_path)
        if not isinstance(raw, (bytes, bytearray)):
            raw = b"".join(raw)
        data = transform_index(bytes(raw), operator=True)
        self._ai_doc_ops_body = data
        return data

    def get_content_size(self: Any) -> int:
        body = _body(self)
        if body is None:
            return orig_size(self)
        return len(body)

    def get_content(
        first: Any,
        abspath: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ):
        # Tornado calls this two ways. An instance call is
        # get_content(handler, path, start, end). The static-url hash calls
        # get_content(path, start, end) on the class, and a plain function
        # does not bind, so the path arrives first. The saved original is a
        # classmethod already bound to the class.
        if not hasattr(first, "request"):
            return orig_content(first, abspath, start)
        if start is None and end is None and abspath == getattr(first, "absolute_path", None):
            body = _body(first)
            if body is not None:
                return body
        return orig_content(abspath, start, end)

    def compute_etag(self: Any):
        etag = orig_etag(self) if orig_etag is not None else None
        if _is_index(self) and _operator(self):
            base = etag or '"index"'
            return f"{base}-ops"
        return etag

    handler_cls.get_content_size = get_content_size
    handler_cls.get_content = get_content
    if orig_etag is not None:
        handler_cls.compute_etag = compute_etag
    handler_cls._ai_doc_ops_installed = True
