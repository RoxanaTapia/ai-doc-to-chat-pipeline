"""Request/response models for the thin chat API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload for `GET /health`."""

    status: str = "ok"
    service: str = "ai-doc-api"


class ChatRequest(BaseModel):
    """Grounded chat request.

    Pass retrieved ``context`` with the ``query``. Session/document persistence
    is out of scope for thin M8 (see M9); optional ``session_id`` is reserved.
    """

    query: str = Field(..., min_length=1, description="User question")
    context: str = Field(
        ...,
        min_length=1,
        description="Retrieved document context used to ground the answer",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional client correlation id (not persisted in thin M8)",
    )
    dummy_mode: bool | None = Field(
        default=None,
        description=(
            "When set, maps to dummy vs ollama if LLM_PROVIDER is unset. "
            "Omit to follow LLM_PROVIDER / server defaults (dummy_mode=True)."
        ),
    )


class ChatResponse(BaseModel):
    """Grounded answer JSON for `POST /chat`."""

    answer: str
    provider: str
    session_id: str | None = None
