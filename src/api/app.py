"""Thin FastAPI app: GET /health, POST /chat, OpenAPI at /docs."""

from __future__ import annotations

from fastapi import FastAPI

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from rag import generate_answer, resolve_llm_provider_name

app = FastAPI(
    title="AI Doc Chat API",
    description=(
        "Thin integration surface for private document Q&A. "
        "Pass retrieved context with each chat request; persistence is optional later."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check for probes and proposal demos."""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(body: ChatRequest) -> ChatResponse:
    """Return a grounded answer using the same LLM providers as the Streamlit app."""
    dummy_mode = True if body.dummy_mode is None else body.dummy_mode
    provider = resolve_llm_provider_name(dummy_mode=dummy_mode)
    answer = generate_answer(
        context=body.context,
        query=body.query,
        dummy_mode=dummy_mode,
    )
    return ChatResponse(
        answer=answer,
        provider=provider,
        session_id=body.session_id,
    )
