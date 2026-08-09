#!/usr/bin/env python3
"""Invite redeem, request-by-email, and forward_auth verify service.

Serves both ai-doc-pilot and receipt-intelligence from a single process.
Site is resolved per-request from the Host header (or explicit ``site`` POST
field, falling back to pilot for backward compatibility).
"""

from __future__ import annotations

import json
import os
import re
import sys
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

sys.path.insert(0, str(Path(__file__).resolve().parent))

from disposable_domains import is_disposable_email  # noqa: E402
from mail import send_email, smtp_configured  # noqa: E402
from rate_limit import RateLimiter  # noqa: E402
from sites import SiteConfig, load_registry, resolve_site  # noqa: E402
from spam_guards import (  # noqa: E402
    honeypot_triggered,
    should_notify_on_redeem,
    should_notify_on_request,
)
from tokens import mint, parse_ttl, verify  # noqa: E402
from turnstile import (  # noqa: E402
    extract_turnstile_token,
    turnstile_enabled,
    verify_turnstile,
)

HOST = os.environ.get("INVITE_BIND", "0.0.0.0")
PORT = int(os.environ.get("INVITE_PORT", "8090"))
COOKIE_MAX_AGE = int(os.environ.get("INVITE_COOKIE_MAX_AGE", str(7 * 86400)))
APP_PATH = os.environ.get("INVITE_APP_PATH", "/app")
REQUEST_TTL = os.environ.get("INVITE_REQUEST_TTL", "72h")
RATE_MAX = int(os.environ.get("INVITE_REQUEST_MAX", "3"))
RATE_WINDOW = int(os.environ.get("INVITE_REQUEST_WINDOW_SECONDS", "3600"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
SUCCESS_MESSAGE = (
    "If that address can receive mail, a temporary invite is on its way. "
    "The invite expires soon—open the link or paste the code under "
    "I have an invite."
)
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)
ip_limiter = RateLimiter(RATE_MAX, RATE_WINDOW)
email_limiter = RateLimiter(RATE_MAX, RATE_WINDOW)

# Loaded once at startup; all threads read (no writes after init).
_REGISTRY: dict[str, SiteConfig] = {}


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


def _structured_log(event: str, **fields: Any) -> None:
    """One JSON line to stderr for invite paper trail (separate from quiet access logs)."""
    payload = {"event": event, **{k: v for k, v in fields.items() if v is not None}}
    sys.stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "InviteAuth/1.4"

    def log_message(self, fmt: str, *args) -> None:
        # Quiet access-style lines for health/verify; structured logs are the paper trail.
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

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

    def _request_meta(self) -> dict[str, str]:
        return {
            "client_ip": _client_ip(self),
            "user_agent": self.headers.get("User-Agent", "")[:500],
            "referer": self.headers.get("Referer", "")[:500],
        }

    def _log_request(
        self,
        *,
        email: str,
        site: str,
        outcome: str,
        http_status: int,
    ) -> None:
        meta = self._request_meta()
        _structured_log(
            "invite_request",
            email=email or None,
            client_ip=meta["client_ip"],
            user_agent=meta["user_agent"] or None,
            referer=meta["referer"] or None,
            site=site,
            outcome=outcome,
            http_status=http_status,
        )

    def _log_redeem(
        self,
        *,
        site: str,
        outcome: str,
        http_status: int,
        label: str | None = None,
    ) -> None:
        meta = self._request_meta()
        _structured_log(
            "invite_redeem",
            email=None,
            label=label or None,
            client_ip=meta["client_ip"],
            user_agent=meta["user_agent"] or None,
            referer=meta["referer"] or None,
            site=site,
            outcome=outcome,
            http_status=http_status,
        )

    def _log_exit(
        self,
        *,
        site: str,
        outcome: str,
        http_status: int,
    ) -> None:
        meta = self._request_meta()
        _structured_log(
            "invite_exit",
            client_ip=meta["client_ip"],
            site=site,
            outcome=outcome,
            http_status=http_status,
        )

    def _resolved_site(self, data: dict | None = None) -> SiteConfig:
        host = self.headers.get("Host", "").strip()
        site_param = (data or {}).get("site") if data else None
        return resolve_site(host, site_param, _REGISTRY)

    def _cookie_header(self, token: str, site_cfg: SiteConfig) -> str:
        return (
            f"{site_cfg.cookie}={token}; Path=/; HttpOnly; Secure; SameSite=Lax; "
            f"Max-Age={COOKIE_MAX_AGE}"
        )

    def _clear_cookie_header(self, site_cfg: SiteConfig) -> str:
        """Expire the site invite cookie (same attributes as set, empty value)."""
        return f"{site_cfg.cookie}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0"

    def _token_from_cookie(self, site_cfg: SiteConfig) -> str | None:
        raw = self.headers.get("Cookie", "")
        if not raw:
            return None
        cookie = SimpleCookie()
        cookie.load(raw)
        morsel = cookie.get(site_cfg.cookie)
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
            site_cfg = self._resolved_site()
            token = self._token_from_cookie(site_cfg)
            if not token:
                # Stale/empty cookie still matches Caddy *invite=*; clear and send
                # to the gate instead of a bare 401 that blocks Basic Auth fallback.
                self._exit(site_cfg, outcome="verify_missing")
                return
            try:
                payload = verify(token)
            except (RuntimeError, ValueError):
                self._exit(site_cfg, outcome="verify_invalid")
                return
            # Token site must match resolved site.
            if payload.get("site", "pilot") != site_cfg.key:
                self._exit(site_cfg, outcome="verify_site_mismatch")
                return
            self._send(200, b"ok\n", "text/plain; charset=utf-8", {"X-Invite-Ok": "1"})
            return

        if path == "/invite/redeem":
            qs = parse_qs(parsed.query)
            token = (qs.get("token") or [""])[0]
            site_cfg = self._resolved_site()
            self._redeem(token, site_cfg)
            return

        if path == "/invite/exit":
            self._exit(self._resolved_site())
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
            site_cfg = self._resolved_site(data)
            self._request_invite(data, site_cfg)
            return

        if path == "/invite/redeem":
            token = str(data.get("token") or data.get("code") or "")
            site_cfg = self._resolved_site(data)
            self._redeem(token, site_cfg)
            return

        if path == "/invite/exit":
            self._exit(self._resolved_site(data))
            return

        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def _request_invite(self, data: dict, site_cfg: SiteConfig) -> None:
        email = str(data.get("email") or "").strip().lower()
        wants_json = "application/json" in self.headers.get("Accept", "")
        site_key = site_cfg.key

        def respond(
            status: int,
            title: str,
            message: str,
            *,
            err: bool = False,
            outcome: str,
        ) -> None:
            self._log_request(
                email=email,
                site=site_key,
                outcome=outcome,
                http_status=status,
            )
            if wants_json:
                payload = json.dumps({"ok": not err, "message": message}).encode()
                self._send(status, payload, "application/json; charset=utf-8")
                return
            body = f'<p class="err">{message}</p>' if err else f"<p>{message}</p>"
            st, html, ctype = _html_page(title, body, status)
            self._send(st, html, ctype)

        # Honeypot first: look like success, mint nothing, tip off nobody.
        if honeypot_triggered(data):
            respond(200, "Check your email", SUCCESS_MESSAGE, outcome="honeypot")
            return

        if not EMAIL_RE.match(email):
            respond(
                400,
                "Check your email",
                "Enter a valid email address.",
                err=True,
                outcome="invalid_email",
            )
            return

        if is_disposable_email(email):
            respond(
                400,
                "Check your email",
                "That email domain is not accepted. Please use a different address.",
                err=True,
                outcome="disposable",
            )
            return

        if turnstile_enabled():
            token = extract_turnstile_token(data)
            if not verify_turnstile(token, remoteip=_client_ip(self)):
                respond(
                    400,
                    "Please try again",
                    "Could not verify this request. Refresh the page and try again.",
                    err=True,
                    outcome="turnstile_failed",
                )
                return

        if not smtp_configured():
            respond(
                503,
                "Invite unavailable",
                "Email delivery is not configured yet. Please try again later.",
                err=True,
                outcome="smtp_unconfigured",
            )
            return

        ip = _client_ip(self)
        if not ip_limiter.allow(f"ip:{ip}") or not email_limiter.allow(f"email:{email}"):
            respond(
                429,
                "Please wait",
                "Too many invite requests. Try again in about an hour.",
                err=True,
                outcome="rate_limited",
            )
            return

        try:
            ttl_seconds = parse_ttl(REQUEST_TTL)
            invite_token = mint(ttl_seconds, label=f"request:{email}", site=site_cfg.key)
        except (RuntimeError, ValueError) as exc:
            respond(503, "Invite unavailable", str(exc), err=True, outcome="mint_failed")
            return

        redeem_url = f"{site_cfg.base_url}/invite/redeem?token={invite_token}"
        ttl_label = _ttl_label(ttl_seconds)
        ctx = {
            "code": invite_token,
            "redeem_url": redeem_url,
            "ttl_label": ttl_label,
            "product_name": site_cfg.product_name,
            "host_label": site_cfg.host_label,
            "access_phrase": site_cfg.access_phrase,
        }
        try:
            html_body = jinja_env.get_template("invite_email.html").render(**ctx)
            text_body = jinja_env.get_template("invite_email.txt").render(**ctx)
            send_email(
                to=email,
                subject=f"Your temporary {site_cfg.product_name} invite",
                text_body=text_body,
                html_body=html_body,
            )
            # INVITE_NOTIFY_ON default is redeem (see spam_guards). Legacy: request.
            if (
                should_notify_on_request()
                and site_cfg.notify_to
                and site_cfg.notify_to.lower() != email
            ):
                send_email(
                    to=site_cfg.notify_to,
                    subject=f"Invite requested: {email} ({site_cfg.host_label})",
                    text_body=(
                        f"{email} requested a temporary {site_cfg.product_name} invite "
                        f"(TTL {ttl_label}).\n"
                    ),
                    html_body=(
                        f"<p><strong>{email}</strong> requested a temporary "
                        f"{site_cfg.product_name} invite (TTL {ttl_label}).</p>"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 — surface as calm 503
            sys.stderr.write(f"invite request email failed: {exc}\n")
            respond(
                503,
                "Invite unavailable",
                "Could not send the invite email. Please try again later.",
                err=True,
                outcome="email_failed",
            )
            return

        respond(200, "Check your email", SUCCESS_MESSAGE, outcome="ok")

    def _exit(self, site_cfg: SiteConfig, *, outcome: str = "ok") -> None:
        """Clear the site invite cookie and return to that host's public home.

        Also used when ``/verify`` fails under Caddy ``forward_auth``: a bare 401
        was shown as the page and blocked the Basic Auth Login fallback.
        """
        self._log_exit(site=site_cfg.key, outcome=outcome, http_status=303)
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie", self._clear_cookie_header(site_cfg))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _notify_redeem(self, site_cfg: SiteConfig, label: str | None) -> None:
        if not should_notify_on_redeem():
            return
        if not site_cfg.notify_to or not smtp_configured():
            return
        who = (label or "unknown").strip() or "unknown"
        subject = f"Invite redeemed: {who} ({site_cfg.host_label})"
        text_body = (
            f"Invite redeemed for {site_cfg.product_name} ({site_cfg.host_label}).\n"
            f"Token label: {who}\n"
        )
        html_body = (
            f"<p>Invite redeemed for <strong>{site_cfg.product_name}</strong> "
            f"({site_cfg.host_label}).</p>"
            f"<p>Token label: <code>{who}</code></p>"
        )
        try:
            send_email(
                to=site_cfg.notify_to,
                subject=subject,
                text_body=text_body,
                html_body=html_body,
            )
        except Exception as exc:  # noqa: BLE001 — redeem already succeeded
            sys.stderr.write(f"invite redeem notify failed: {exc}\n")

    def _redeem(self, token: str, site_cfg: SiteConfig) -> None:
        token = (token or "").strip()
        site_key = site_cfg.key
        if not token:
            self._log_redeem(
                site=site_key,
                outcome="redeem_missing",
                http_status=400,
            )
            status, body, ctype = _html_page(
                "Invite needed",
                '<p class="err">Paste an invite code, or open the signed link from your email.</p>',
                400,
            )
            self._send(status, body, ctype)
            return
        try:
            payload = verify(token)
        except RuntimeError as exc:
            self._log_redeem(
                site=site_key,
                outcome="redeem_unavailable",
                http_status=503,
            )
            status, body, ctype = _html_page("Invite unavailable", f'<p class="err">{exc}</p>', 503)
            self._send(status, body, ctype)
            return
        except ValueError as exc:
            self._log_redeem(
                site=site_key,
                outcome="redeem_invalid",
                http_status=401,
            )
            status, body, ctype = _html_page("Invite invalid", f'<p class="err">{exc}</p>', 401)
            self._send(status, body, ctype)
            return

        # Token site claim must match the resolved site (old tokens default to pilot).
        if payload.get("site", "pilot") != site_cfg.key:
            self._log_redeem(
                site=site_key,
                outcome="redeem_site_mismatch",
                http_status=401,
                label=str(payload.get("label") or "") or None,
            )
            status, body, ctype = _html_page(
                "Invite invalid",
                '<p class="err">This invite is not valid for this site.</p>',
                401,
            )
            self._send(status, body, ctype)
            return

        label = str(payload.get("label") or "") or None
        self._notify_redeem(site_cfg, label)
        self._log_redeem(
            site=site_key,
            outcome="redeem_ok",
            http_status=303,
            label=label,
        )

        self.send_response(303)
        self.send_header("Location", APP_PATH)
        self.send_header("Set-Cookie", self._cookie_header(token, site_cfg))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def main() -> int:
    global _REGISTRY
    try:
        from tokens import get_secret

        get_secret()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _REGISTRY = load_registry()

    if not smtp_configured():
        print(
            "warning: SMTP_HOST/SMTP_FROM not set; /invite/request will return 503",
            flush=True,
        )
    if turnstile_enabled():
        print("turnstile: enabled (siteverify before mint)", flush=True)
    else:
        print("turnstile: disabled (local smoke / TURNSTILE_ENABLED=false)", flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"invite auth listening on {HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
