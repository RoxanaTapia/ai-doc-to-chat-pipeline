"""Optional HTTP client: Streamlit → thin FastAPI when API_BASE_URL is set."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import httpx

logger = logging.getLogger(__name__)


def api_base_url() -> str | None:
    """Return stripped API base URL, or None when Streamlit should generate in-process."""
    raw = (os.getenv("API_BASE_URL") or "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


def chat_via_api(
    *,
    context: str,
    query: str,
    dummy_mode: bool = True,
    session_id: str | None = None,
    timeout_seconds: float = 120.0,
) -> str:
    """POST /chat and return the answer string. Raises on transport/HTTP errors."""
    base = api_base_url()
    if not base:
        raise RuntimeError("API_BASE_URL is not set")

    payload = {
        "query": query,
        "context": context,
        "dummy_mode": dummy_mode,
    }
    if session_id:
        payload["session_id"] = session_id

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{base}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise RuntimeError("API /chat response missing string 'answer'")
    return answer


def chat_via_api_stream(
    *,
    context: str,
    query: str,
    dummy_mode: bool = True,
    session_id: str | None = None,
    timeout_seconds: float = 120.0,
) -> Iterator[str]:
    """Yield the API answer as one chunk (thin /chat is non-streaming today)."""
    answer = chat_via_api(
        context=context,
        query=query,
        dummy_mode=dummy_mode,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
    )
    if answer:
        yield answer
