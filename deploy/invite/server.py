#!/usr/bin/env python3
"""Invite redeem, request-by-email, and forward_auth verify service."""

from __future__ import annotations

import json
import os
import re
import sys
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mail import send_email, smtp_configured  # noqa: E402
from rate_limit import RateLimiter  # noqa: E402
from tokens import COOKIE_NAME, mint, parse_ttl, verify  # noqa: E402

HOST = os.environ.get("INVITE_BIND", "0.0.0.0")
PORT = int(os.environ.get("INVITE_PORT", "8090"))
COOKIE_MAX_AGE = int(os.environ.get("INVITE_COOKIE_MAX_AGE", str(7 * 86400)))
APP_PATH = os.environ.get("INVITE_APP_PATH", "/app")
BASE_URL = os.environ.get(
    "INVITE_BASE_URL", "https://ai-doc-pilot.roxanatapia.dev"
).rstrip("/")
REQUEST_TTL = os.environ.get("INVITE_REQUEST_TTL", "72h")
NOTIFY_TO = os.environ.get("INVITE_NOTIFY_TO", "hello@roxanatapia.dev").strip()
RATE_MAX = int(os.environ.get("INVITE_REQUEST_MAX", "3"))
RATE_WINDOW = int(os.environ.get("INVITE_REQUEST_WINDOW_SECONDS", "3600"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)
ip_limiter = RateLimiter(RATE_MAX, RATE_WINDOW)
email_limiter = RateLimiter(RATE_MAX, RATE_WINDOW)


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


def _ttl_label(ttl_seconds: int) -> str:
    if ttl_seconds % 86400 == 0:
        days = ttl_seconds // 86400
        return f"{days} day" if days == 1 else f"{days} days"
    if ttl_seconds % 3600 == 0:
        hours = ttl_seconds // 3600
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{ttl_seconds} seconds"


def _client_ip(handler: BaseHTTPRequestHandler) -> str:
    forwarded = handler.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    real = handler.headers.get("X-Real-IP", "").strip()
    if real:
        return real
    return handler.client_address[0]


class Handler(BaseHTTPRequestHandler):
    server_version = "InviteAuth/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
    ) -> None:
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
            f"{COOKIE_NAME}={token}; Path=/; HttpOnly; Secure; SameSite=Lax; "
            f"Max-Age={COOKIE_MAX_AGE}"
        )

    def _token_from_cookie(self) -> str | None:
        raw = self.headers.get("Cookie", "")
        if not raw:
            return None
        cookie = SimpleCookie()
        cookie.load(raw)
        morsel = cookie.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def _read_body(self) -> tuple[str, dict]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
        content_type = self.headers.get("Content-Type", "")
        data: dict = {}
        if "application/json" in content_type:
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {}
        else:
            form = parse_qs(raw)
            data = {k: (v[0] if v else "") for k, v in form.items()}
        return raw, data

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/healthz":
            self._send(200, b"ok\n", "text/plain; charset=utf-8")
            return

        if path == "/verify":
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
            "<p>Use the gate to <strong>Request an invite</strong> or "
            "<strong>I have an invite</strong>.</p>",
        )
        self._send(status, body, ctype)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        _, data = self._read_body()

        if path == "/invite/request":
            self._request_invite(str(data.get("email") or ""))
            return

        if path == "/invite/redeem":
            token = str(data.get("token") or data.get("code") or "")
            self._redeem(token)
            return

        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _request_invite(self, email: str) -> None:
        email = (email or "").strip().lower()
        wants_json = "application/json" in self.headers.get("Accept", "")

        def respond(status: int, title: str, message: str, err: bool = False) -> None:
            if wants_json:
                payload = json.dumps({"ok": not err, "message": message}).encode()
                self._send(status, payload, "application/json; charset=utf-8")
                return
            body = f'<p class="err">{message}</p>' if err else f"<p>{message}</p>"
            st, html, ctype = _html_page(title, body, status)
            self._send(st, html, ctype)

        if not EMAIL_RE.match(email):
            respond(400, "Check your email", "Enter a valid email address.", err=True)
            return

        if not smtp_configured():
            respond(
                503,
                "Invite unavailable",
                "Email delivery is not configured yet. Please try again later.",
                err=True,
            )
            return

        ip = _client_ip(self)
        if not ip_limiter.allow(f"ip:{ip}") or not email_limiter.allow(f"email:{email}"):
            respond(
                429,
                "Please wait",
                "Too many invite requests. Try again in about an hour.",
                err=True,
            )
            return

        try:
            ttl_seconds = parse_ttl(REQUEST_TTL)
            token = mint(ttl_seconds, label=f"request:{email}")
        except (RuntimeError, ValueError) as exc:
            respond(503, "Invite unavailable", str(exc), err=True)
            return

        redeem_url = f"{BASE_URL}/invite/redeem?token={token}"
        ttl_label = _ttl_label(ttl_seconds)
        ctx = {"code": token, "redeem_url": redeem_url, "ttl_label": ttl_label}
        try:
            html_body = jinja_env.get_template("invite_email.html").render(**ctx)
            text_body = jinja_env.get_template("invite_email.txt").render(**ctx)
            send_email(
                to=email,
                subject="Your temporary AI Doc invite",
                text_body=text_body,
                html_body=html_body,
            )
            if NOTIFY_TO and NOTIFY_TO.lower() != email:
                send_email(
                    to=NOTIFY_TO,
                    subject=f"Invite requested: {email}",
                    text_body=(
                        f"{email} requested a temporary AI Doc invite "
                        f"(TTL {ttl_label}).\n"
                    ),
                    html_body=(
                        f"<p><strong>{email}</strong> requested a temporary "
                        f"AI Doc invite (TTL {ttl_label}).</p>"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 — surface as calm 503
            sys.stderr.write(f"invite request email failed: {exc}\n")
            respond(
                503,
                "Invite unavailable",
                "Could not send the invite email. Please try again later.",
                err=True,
            )
            return

        respond(
            200,
            "Check your email",
            "If that address can receive mail, a temporary invite is on its way. "
            "The invite expires soon—open the link or paste the code under "
            "I have an invite.",
        )

    def _redeem(self, token: str) -> None:
        token = (token or "").strip()
        if not token:
            status, body, ctype = _html_page(
                "Invite needed",
                '<p class="err">Paste an invite code, or open the signed link '
                "from your email.</p>",
                400,
            )
            self._send(status, body, ctype)
            return
        try:
            verify(token)
        except RuntimeError as exc:
            status, body, ctype = _html_page(
                "Invite unavailable", f'<p class="err">{exc}</p>', 503
            )
            self._send(status, body, ctype)
            return
        except ValueError as exc:
            status, body, ctype = _html_page(
                "Invite invalid", f'<p class="err">{exc}</p>', 401
            )
            self._send(status, body, ctype)
            return

        self.send_response(303)
        self.send_header("Location", APP_PATH)
        self.send_header("Set-Cookie", self._cookie_header(token))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def main() -> int:
    try:
        from tokens import get_secret

        get_secret()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not smtp_configured():
        print(
            "warning: SMTP_HOST/SMTP_FROM not set; /invite/request will return 503",
            flush=True,
        )

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"invite auth listening on {HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
