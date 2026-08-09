"""Honeypot + operator-notify policy helpers for invite spam hardening."""

from __future__ import annotations

import os

# Default: notify the operator only after a successful redeem (cookie set),
# not on every /invite/request. That cuts inbox noise from bots that never
# open the app. Override with INVITE_NOTIFY_ON=request (legacy) or none.
_VALID_NOTIFY = frozenset({"redeem", "request", "none"})


def honeypot_triggered(data: dict) -> bool:
    """True when a hidden honeypot field was filled (bots often autofill these)."""
    for key in ("company", "website"):
        if str(data.get(key) or "").strip():
            return True
    return False


def invite_notify_on() -> str:
    """Return redeem | request | none (default redeem)."""
    raw = os.environ.get("INVITE_NOTIFY_ON", "redeem").strip().lower() or "redeem"
    return raw if raw in _VALID_NOTIFY else "redeem"


def should_notify_on_request() -> bool:
    return invite_notify_on() == "request"


def should_notify_on_redeem() -> bool:
    return invite_notify_on() == "redeem"
