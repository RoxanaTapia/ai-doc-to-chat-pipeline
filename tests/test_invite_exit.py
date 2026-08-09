"""Tests for GET/POST /invite/exit (clear cookie → public home) (#132)."""

from __future__ import annotations

import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

INVITE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "invite"
if str(INVITE_DIR) not in sys.path:
    sys.path.insert(0, str(INVITE_DIR))


@pytest.fixture
def invite_server(monkeypatch: pytest.MonkeyPatch):
    """Start invite Handler on an ephemeral port with a loaded site registry."""
    monkeypatch.setenv("INVITE_SECRET", "test-invite-secret-for-exit")
    monkeypatch.setenv("INVITE_BASE_URL", "https://ai-doc-pilot.roxanatapia.dev")
    monkeypatch.setenv(
        "RECEIPT_INVITE_BASE_URL",
        "https://receipt-intelligence.roxanatapia.dev",
    )

    import server as invite_server_mod
    from sites import load_registry

    invite_server_mod._REGISTRY = load_registry()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), invite_server_mod.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield {"host": host, "port": port, "mod": invite_server_mod}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _request(
    invite_server: dict,
    *,
    method: str,
    path: str,
    host: str,
    cookie: str | None = None,
) -> tuple[int, dict[str, str]]:
    conn = HTTPConnection(invite_server["host"], invite_server["port"], timeout=5)
    try:
        headers = {"Host": host}
        if cookie:
            headers["Cookie"] = cookie
        conn.request(method, path, headers=headers)
        resp = conn.getresponse()
        headers_out = {k.lower(): v for k, v in resp.getheaders()}
        # Drain body so connection can close cleanly.
        resp.read()
        return resp.status, headers_out
    finally:
        conn.close()


def _assert_exit_clears(
    status: int,
    headers: dict[str, str],
    *,
    cookie_name: str,
) -> None:
    assert status == 303
    assert headers.get("location") == "/"
    set_cookie = headers.get("set-cookie", "")
    assert set_cookie.startswith(f"{cookie_name}=")
    assert "Max-Age=0" in set_cookie
    assert "Path=/" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=Lax" in set_cookie
    # Empty cookie value: name=; or name= followed by attribute separator.
    prefix = f"{cookie_name}="
    after = set_cookie[len(prefix) :]
    assert after.startswith(";") or after.startswith(" ;")


def test_exit_get_pilot_default_host(invite_server: dict) -> None:
    status, headers = _request(
        invite_server,
        method="GET",
        path="/invite/exit",
        host="ai-doc-pilot.roxanatapia.dev",
    )
    _assert_exit_clears(status, headers, cookie_name="pilot_invite")


def test_exit_get_pilot_unknown_host_defaults(invite_server: dict) -> None:
    """Unrecognized Host falls back to pilot cookie (resolve_site default)."""
    status, headers = _request(
        invite_server,
        method="GET",
        path="/invite/exit",
        host="localhost",
    )
    _assert_exit_clears(status, headers, cookie_name="pilot_invite")


def test_exit_get_receipt_host(invite_server: dict) -> None:
    status, headers = _request(
        invite_server,
        method="GET",
        path="/invite/exit",
        host="receipt-intelligence.roxanatapia.dev",
    )
    _assert_exit_clears(status, headers, cookie_name="receipt_invite")


def test_exit_post_also_clears(invite_server: dict) -> None:
    status, headers = _request(
        invite_server,
        method="POST",
        path="/invite/exit",
        host="receipt-intelligence.roxanatapia.dev",
    )
    _assert_exit_clears(status, headers, cookie_name="receipt_invite")


def test_clear_cookie_header_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVITE_SECRET", "test-invite-secret-for-exit")
    import server as invite_server_mod
    from sites import SiteConfig

    cfg = SiteConfig(
        key="pilot",
        cookie="pilot_invite",
        base_url="https://example.test",
        product_name="AI Doc",
        host_label="AI Doc",
        access_phrase="by invitation",
        notify_to="ops@example.test",
    )
    # Instance method does not use self; exercise the Set-Cookie shape directly.
    header = invite_server_mod.Handler._clear_cookie_header(object(), cfg)  # type: ignore[arg-type]
    assert header == "pilot_invite=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0"


def test_verify_missing_cookie_redirects_home(invite_server: dict) -> None:
    """Empty/stale cookie name still matches Caddy; verify must not 401 the page."""
    status, headers = _request(
        invite_server,
        method="GET",
        path="/verify",
        host="ai-doc-pilot.roxanatapia.dev",
        cookie="pilot_invite=",
    )
    _assert_exit_clears(status, headers, cookie_name="pilot_invite")


def test_verify_invalid_token_redirects_home(invite_server: dict) -> None:
    status, headers = _request(
        invite_server,
        method="GET",
        path="/verify",
        host="ai-doc-pilot.roxanatapia.dev",
        cookie="pilot_invite=not-a-valid-token",
    )
    _assert_exit_clears(status, headers, cookie_name="pilot_invite")


def test_verify_valid_token_still_ok(invite_server: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVITE_SECRET", "test-invite-secret-for-exit")
    from tokens import mint

    token = mint(3600, label="smoke", site="pilot")
    status, headers = _request(
        invite_server,
        method="GET",
        path="/verify",
        host="ai-doc-pilot.roxanatapia.dev",
        cookie=f"pilot_invite={token}",
    )
    assert status == 200
    assert headers.get("x-invite-ok") == "1"
