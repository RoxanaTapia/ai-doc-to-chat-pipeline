"""Cloudflare Turnstile server-side verification (stdlib urllib)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
DEFAULT_TIMEOUT_S = 5.0


def turnstile_secret() -> str:
    return os.environ.get("TURNSTILE_SECRET_KEY", "").strip()


def turnstile_enabled() -> bool:
    """Enabled by default when TURNSTILE_SECRET_KEY is set; opt out with TURNSTILE_ENABLED=false."""
    explicit = os.environ.get("TURNSTILE_ENABLED", "").strip().lower()
    if explicit in ("0", "false", "no"):
        return False
    if explicit in ("1", "true", "yes"):
        return True
    return bool(turnstile_secret())


def extract_turnstile_token(data: dict) -> str:
    """Accept Cloudflare's standard form field or JSON aliases."""
    return str(
        data.get("cf-turnstile-response")
        or data.get("turnstile_token")
        or data.get("cf_turnstile_response")
        or ""
    ).strip()


def verify_turnstile(
    token: str,
    *,
    remoteip: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    """POST siteverify; return True only when Cloudflare reports success.

    Fail closed on missing secret, empty token, HTTP errors, or malformed JSON.
    """
    secret = turnstile_secret()
    if not secret or not token:
        return False

    form: dict[str, str] = {"secret": secret, "response": token}
    if remoteip:
        form["remoteip"] = remoteip

    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        SITEVERIFY_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False

    return bool(isinstance(payload, dict) and payload.get("success") is True)
