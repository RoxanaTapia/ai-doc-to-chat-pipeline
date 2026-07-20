"""Tests for optional Streamlit → FastAPI client."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api_client  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)


def test_api_base_url_unset() -> None:
    assert api_client.api_base_url() is None


def test_api_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")
    assert api_client.api_base_url() == "http://api:8000"


def test_chat_via_api_posts_and_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://api.test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        body = request.read()
        assert b"notice period" in body
        return httpx.Response(200, json={"answer": "30 days", "provider": "dummy"})

    transport = httpx.MockTransport(handler)

    class _Client(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(api_client.httpx, "Client", _Client)
    answer = api_client.chat_via_api(
        context="Section A: 30 days notice.",
        query="What is the notice period?",
        dummy_mode=True,
    )
    assert answer == "30 days"


def test_chat_via_api_stream_yields_one_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://api.test")
    monkeypatch.setattr(
        api_client,
        "chat_via_api",
        lambda **kwargs: "full answer",
    )
    assert list(
        api_client.chat_via_api_stream(context="ctx", query="q", dummy_mode=True)
    ) == ["full answer"]


def test_chat_via_api_requires_base_url() -> None:
    with pytest.raises(RuntimeError, match="API_BASE_URL"):
        api_client.chat_via_api(context="ctx", query="q")
