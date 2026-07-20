"""Thin FastAPI /health and /chat contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_llm_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai-doc-api"}


def test_chat_dummy_returns_placeholder(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "query": "What is the notice period?",
            "context": "Section A says notice period is 30 days.",
            "dummy_mode": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "dummy"
    assert "no AI model is running here" in payload["answer"]
    assert payload["session_id"] is None


def test_chat_echoes_session_id(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={
            "query": "Hello?",
            "context": "Some context for grounding.",
            "session_id": "sess-1",
            "dummy_mode": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "sess-1"


def test_chat_rejects_empty_query(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"query": "", "context": "Some context", "dummy_mode": True},
    )
    assert response.status_code == 422


def test_chat_llm_provider_env_wins(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    response = client.post(
        "/chat",
        json={
            "query": "What applies?",
            "context": "Clause text",
            "dummy_mode": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "dummy"


def test_openapi_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/chat" in paths
