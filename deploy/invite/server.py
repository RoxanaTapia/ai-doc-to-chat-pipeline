#!/usr/bin/env python3
"""Tiny invite redeem + forward_auth verify service (stdlib only)."""

from __future__ import annotations

import os
import sys
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokens import COOKIE_NAME, verify  # noqa: E402

HOST = os.environ.get("INVITE_BIND", "0.0.0.0")
PORT = int(os.environ.get("INVITE_PORT", "8090"))
COOKIE_MAX_AGE = int(os.environ.get("INVITE_COOKIE_MAX_AGE", str(7 * 86400)))
APP_PATH = os.environ.get("INVITE_APP_PATH", "/app")


def _html_page(title: str, body: str, status: int = 200) -> tuple[int, bytes, str]:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0c1218; color: #e8eef0;
           max-width: 28rem; margin: 4rem auto; padding: 0 1.25rem; line-height: 1.5; }}
    a {{ color: #7eb8c0; }}
    .err {{ color: #f0a8a8; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {body}
  <p><a href="/">Back to gate</a></p>
</body>
</html>
"""
    return status, html.encode("utf-8"), "text/html; charset=utf-8"


class Handler(BaseHTTPRequestHandler):
    server_version = "InviteAuth/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _cookie_header(self, token: str) -> str:
        return (
            f"{COOKIE_NAME}={token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age={COOKIE_MAX_AGE}"
        )

    def _token_from_cookie(self) -> str | None:
        raw = self.headers.get("Cookie", "")
        if not raw:
            return None
        cookie = SimpleCookie()
        cookie.load(raw)
        morsel = cookie.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/healthz":
            self._send(200, b"ok\n", "text/plain; charset=utf-8")
            return

        if path == "/verify":
            # Caddy forward_auth
            token = self._token_from_cookie()
            if not token:
                self._send(401, b"missing invite\n", "text/plain; charset=utf-8")
                return
            try:
                verify(token)
            except (RuntimeError, ValueError):
                self._send(401, b"invalid invite\n", "text/plain; charset=utf-8")
                return
            self._send(200, b"ok\n", "text/plain; charset=utf-8", {"X-Invite-Ok": "1"})
            return

        if path == "/invite/redeem":
            qs = parse_qs(parsed.query)
            token = (qs.get("token") or [""])[0]
            self._redeem(token)
            return

        status, body, ctype = _html_page(
            "Invite",
            "<p>Use the gate’s <strong>Have invite</strong> form, or open a signed redeem URL.</p>",
        )
        self._send(status, body, ctype)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path != "/invite/redeem":
            self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        content_type = self.headers.get("Content-Type", "")
        token = ""
        if "application/json" in content_type:
            import json

            try:
                data = json.loads(raw) if raw else {}
                token = str(data.get("token") or data.get("code") or "")
            except json.JSONDecodeError:
                token = ""
        else:
            form = parse_qs(raw)
            token = (form.get("token") or form.get("code") or [""])[0]
        self._redeem(token)

    def _redeem(self, token: str) -> None:
        token = (token or "").strip()
        if not token:
            status, body, ctype = _html_page(
                "Invite needed",
                '<p class="err">Paste an invite code, or open the signed link from your email.</p>',
                400,
            )
            self._send(status, body, ctype)
            return
        try:
            verify(token)
        except RuntimeError as exc:
            status, body, ctype = _html_page("Invite unavailable", f'<p class="err">{exc}</p>', 503)
            self._send(status, body, ctype)
            return
        except ValueError as exc:
            status, body, ctype = _html_page("Invite invalid", f'<p class="err">{exc}</p>', 401)
            self._send(status, body, ctype)
            return

        # 303 so refresh does not re-POST
        self.send_response(303)
        self.send_header("Location", APP_PATH)
        self.send_header("Set-Cookie", self._cookie_header(token))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def main() -> int:
    try:
        # Fail fast if secret missing
        from tokens import get_secret

        get_secret()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"invite auth listening on {HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
