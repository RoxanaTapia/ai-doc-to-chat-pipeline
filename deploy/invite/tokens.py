"""HMAC-signed, time-limited invite tokens (stdlib only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any


COOKIE_NAME = "pilot_invite"
TOKEN_PREFIX = "v1"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def get_secret() -> bytes:
    secret = os.environ.get("INVITE_SECRET", "").strip()
    if not secret:
        raise RuntimeError("INVITE_SECRET is not set")
    return secret.encode("utf-8")


def mint(ttl_seconds: int, label: str = "") -> str:
    if ttl_seconds < 60:
        raise ValueError("ttl_seconds must be at least 60")
    payload: dict[str, Any] = {
        "exp": int(time.time()) + ttl_seconds,
        "jti": secrets.token_urlsafe(12),
    }
    if label:
        payload["label"] = label
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64url(hmac.new(get_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{TOKEN_PREFIX}.{body}.{sig}"


def verify(token: str) -> dict[str, Any]:
    token = (token or "").strip()
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        raise ValueError("invalid token format")
    body, sig = parts[1], parts[2]
    expected = _b64url(hmac.new(get_secret(), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid signature")
    try:
        payload = json.loads(_b64url_decode(body))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid payload") from exc
    exp = int(payload.get("exp", 0))
    if exp < int(time.time()):
        raise ValueError("token expired")
    return payload


def parse_ttl(text: str) -> int:
    """Parse 24h, 72h, 7d, or bare seconds."""
    text = text.strip().lower()
    if text.endswith("d"):
        return int(float(text[:-1]) * 86400)
    if text.endswith("h"):
        return int(float(text[:-1]) * 3600)
    if text.endswith("m"):
        return int(float(text[:-1]) * 60)
    return int(text)
