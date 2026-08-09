"""Unit tests for invite spam-hardening helpers (#124)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

INVITE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "invite"
if str(INVITE_DIR) not in sys.path:
    sys.path.insert(0, str(INVITE_DIR))

from disposable_domains import is_disposable_email  # noqa: E402
from spam_guards import (  # noqa: E402
    honeypot_triggered,
    invite_notify_on,
    should_notify_on_redeem,
    should_notify_on_request,
)
from turnstile import extract_turnstile_token, turnstile_enabled  # noqa: E402


@pytest.mark.parametrize(
    "email",
    [
        "bot@mailinator.com",
        "x@guerrillamail.com",
        "a@yopmail.com",
        "user@mail.yopmail.com",
        "spam@temp-mail.org",
        "12345678901234@example.com",  # long all-digit local
    ],
)
def test_disposable_email_rejected(email: str) -> None:
    assert is_disposable_email(email) is True


@pytest.mark.parametrize(
    "email",
    [
        "person@example.com",
        "roxana@roxanatapia.dev",
        "ok@company.co.uk",
    ],
)
def test_normal_email_accepted(email: str) -> None:
    assert is_disposable_email(email) is False


def test_honeypot_empty_ok() -> None:
    assert honeypot_triggered({"email": "a@b.com"}) is False
    assert honeypot_triggered({"company": "", "website": "  "}) is False


def test_honeypot_company_or_website() -> None:
    assert honeypot_triggered({"company": "Acme"}) is True
    assert honeypot_triggered({"website": "https://spam.example"}) is True


def test_notify_policy_default_redeem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVITE_NOTIFY_ON", raising=False)
    assert invite_notify_on() == "redeem"
    assert should_notify_on_redeem() is True
    assert should_notify_on_request() is False


def test_notify_policy_request_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVITE_NOTIFY_ON", "request")
    assert should_notify_on_request() is True
    assert should_notify_on_redeem() is False


def test_notify_policy_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVITE_NOTIFY_ON", "none")
    assert should_notify_on_request() is False
    assert should_notify_on_redeem() is False


def test_turnstile_enabled_defaults_with_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURNSTILE_ENABLED", raising=False)
    monkeypatch.setenv("TURNSTILE_SECRET_KEY", "sekrit")
    assert turnstile_enabled() is True
    monkeypatch.setenv("TURNSTILE_ENABLED", "false")
    assert turnstile_enabled() is False


def test_turnstile_disabled_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TURNSTILE_ENABLED", raising=False)
    monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
    assert turnstile_enabled() is False


def test_extract_turnstile_token_aliases() -> None:
    assert extract_turnstile_token({"cf-turnstile-response": "abc"}) == "abc"
    assert extract_turnstile_token({"turnstile_token": "xyz"}) == "xyz"
    assert extract_turnstile_token({}) == ""
