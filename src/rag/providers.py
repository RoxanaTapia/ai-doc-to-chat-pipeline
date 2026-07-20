"""Swappable LLM backends for grounded answer generation."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from rag.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    PROMPTS_PATH,
    load_generation_config,
    load_rag_prompt,
)

logger = logging.getLogger(__name__)

SUPPORTED_LLM_PROVIDERS = frozenset({"ollama", "anthropic", "openai", "dummy"})
DUMMY_UI_RESPONSE = (
    "This is a UI demo — no AI model is running here.\n\n"
    "Upload and search work, but answers are not generated on this host.\n\n"
    "For real grounded answers on your documents, request access to the "
    "[live pilot](https://ai-doc-pilot.roxanatapia.dev)."
)


@runtime_checkable
class LLMProvider(Protocol):
    """Swappable generation backend used by `generate_answer` / `generate_answer_stream`."""

    def generate(self, context: str, query: str) -> str:
        """Return an answer string grounded in `context` for `query`."""
        ...

    def stream(self, context: str, query: str) -> Iterator[str]:
        """Yield answer text chunks for progressive UI (e.g. `st.write_stream`)."""
        ...


def _format_prompt(template: str, context: str, query: str) -> str:
    """Render prompt template while supporting `question` and legacy `query` keys."""
    try:
        # Provide both keys so mixed templates using {question} and {query} work.
        return template.format(context=context, question=query, query=query)
    except (KeyError, ValueError) as exc:
        logger.warning("Invalid prompt template; falling back to default template: %s", exc)
        return DEFAULT_PROMPT_TEMPLATE.format(context=context, question=query, query=query)


def _dummy_response() -> str:
    """Placeholder text for UI-demo / Streamlit Cloud (no local LLM)."""
    return DUMMY_UI_RESPONSE


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content (str or content blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content)


def _prompt_for_generation(context: str, query: str) -> str:
    """Build the RAG prompt from YAML template (or default)."""
    prompt_template = load_rag_prompt() if PROMPTS_PATH.exists() else DEFAULT_PROMPT_TEMPLATE
    return _format_prompt(prompt_template, context=context, query=query)


def _build_ollama_llm(settings: dict[str, Any]) -> ChatOllama:
    """Construct a ChatOllama client from generation settings."""
    effective_temperature = settings["temperature"] if settings["do_sample"] else 0.0
    return ChatOllama(
        model=settings["model"],
        temperature=effective_temperature,
        top_p=settings["top_p"],
        num_predict=settings["max_new_tokens"],
        num_ctx=settings["num_ctx"],
        timeout=settings["timeout_seconds"],
    )


def _build_anthropic_llm(settings: dict[str, Any], api_key: str) -> ChatAnthropic:
    """Construct a ChatAnthropic client from generation settings.

    Haiku 4.5+ rejects requests that set both ``temperature`` and ``top_p``.
    Prefer temperature only (deterministic 0.0 when ``do_sample`` is false).
    """
    effective_temperature = settings["temperature"] if settings["do_sample"] else 0.0
    model = str(settings.get("anthropic_model") or DEFAULT_ANTHROPIC_MODEL)
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=effective_temperature,
        max_tokens=int(settings["max_new_tokens"]),
        timeout=float(settings["timeout_seconds"]),
    )


def _iter_llm_text_chunks(llm: Any, prompt: str) -> Iterator[str]:
    """Yield non-empty text pieces from `llm.stream(prompt)`."""
    for chunk in llm.stream(prompt):
        text = _content_to_text(chunk.content if hasattr(chunk, "content") else chunk)
        if text:
            yield text


def _generate_with_ollama(context: str, query: str, settings: dict[str, Any] | None = None) -> str:
    """Generate answer text using local Ollama runtime."""
    if settings is None:
        settings = load_generation_config()
    prompt = _prompt_for_generation(context, query)
    response = _build_ollama_llm(settings).invoke(prompt)
    return _content_to_text(response.content if hasattr(response, "content") else response).strip()


def _stream_with_ollama(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer text chunks from local Ollama (`ChatOllama.stream`)."""
    if settings is None:
        settings = load_generation_config()
    prompt = _prompt_for_generation(context, query)
    yield from _iter_llm_text_chunks(_build_ollama_llm(settings), prompt)


def _require_anthropic_api_key() -> str:
    """Return ANTHROPIC_API_KEY or raise a clear operator-facing error."""
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your local .env "
            "(never commit the key). Required when LLM_PROVIDER=anthropic."
        )
    return key


def _generate_with_anthropic(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> str:
    """Generate answer text using Anthropic Claude (ChatAnthropic)."""
    if settings is None:
        settings = load_generation_config()
    api_key = _require_anthropic_api_key()
    prompt = _prompt_for_generation(context, query)
    response = _build_anthropic_llm(settings, api_key).invoke(prompt)
    return _content_to_text(response.content if hasattr(response, "content") else response).strip()


def _stream_with_anthropic(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer text chunks from Anthropic (`ChatAnthropic.stream`)."""
    if settings is None:
        settings = load_generation_config()
    api_key = _require_anthropic_api_key()
    prompt = _prompt_for_generation(context, query)
    yield from _iter_llm_text_chunks(_build_anthropic_llm(settings, api_key), prompt)


def _stream_with_generate_fallback(
    stream_fn,
    generate_fn,
    *,
    label: str,
) -> Iterator[str]:
    """Yield from `stream_fn`; if stream fails before any chunk, yield `generate_fn()` once."""
    yielded_any = False
    try:
        for chunk in stream_fn():
            yielded_any = True
            yield chunk
    except Exception as exc:
        if yielded_any:
            logger.exception("%s streaming interrupted after partial output: %s", label, exc)
            raise
        logger.warning("%s streaming failed; falling back to generate(): %s", label, exc)
        answer = generate_fn()
        if answer:
            yield answer


class DummyLLMProvider:
    """UI-demo generator: fixed placeholder, no model calls."""

    def generate(self, context: str, query: str) -> str:
        return _dummy_response()

    def stream(self, context: str, query: str) -> Iterator[str]:
        yield _dummy_response()


class OllamaLLMProvider:
    """Local Ollama generator with optional dummy fallback on errors."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings

    def generate(self, context: str, query: str) -> str:
        settings = self._settings if self._settings is not None else load_generation_config()
        try:
            return _generate_with_ollama(context=context, query=query, settings=settings)
        except Exception as exc:
            logger.exception("Ollama generation failed: %s", exc)
            if settings.get("fallback_to_dummy_on_error", False):
                return f"Ollama unavailable, falling back to dummy mode.\n\n{_dummy_response()}"
            raise RuntimeError(f"Ollama generation unavailable: {exc}") from exc

    def stream(self, context: str, query: str) -> Iterator[str]:
        settings = self._settings if self._settings is not None else load_generation_config()
        yield from _stream_with_generate_fallback(
            lambda: _stream_with_ollama(context=context, query=query, settings=settings),
            lambda: self.generate(context, query),
            label="Ollama",
        )


class AnthropicLLMProvider:
    """Anthropic Claude generator (Haiku default) for fast demo / video recording."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings

    def generate(self, context: str, query: str) -> str:
        settings = self._settings if self._settings is not None else load_generation_config()
        try:
            return _generate_with_anthropic(context=context, query=query, settings=settings)
        except Exception as exc:
            # Missing-key RuntimeError already has a clear message; keep it readable.
            if isinstance(exc, RuntimeError) and "ANTHROPIC_API_KEY" in str(exc):
                raise
            logger.exception("Anthropic generation failed: %s", exc)
            raise RuntimeError(f"Anthropic generation unavailable: {exc}") from exc

    def stream(self, context: str, query: str) -> Iterator[str]:
        settings = self._settings if self._settings is not None else load_generation_config()
        yield from _stream_with_generate_fallback(
            lambda: _stream_with_anthropic(context=context, query=query, settings=settings),
            lambda: self.generate(context, query),
            label="Anthropic",
        )


def resolve_llm_provider_name(dummy_mode: bool = True) -> str:
    """Pick provider name from env, else map legacy dummy_mode / USE_DUMMY_GENERATOR.

    Resolution:
    1. `LLM_PROVIDER` env (non-empty) wins
    2. Else `dummy` if `dummy_mode` else `ollama`
    """
    raw = os.getenv("LLM_PROVIDER")
    if raw is not None and (normalized := raw.strip()):
        return normalized.lower()
    return "dummy" if dummy_mode else "ollama"


def get_llm_provider(
    name: str,
    settings: dict[str, Any] | None = None,
) -> LLMProvider:
    """Factory for generation backends selected by `LLM_PROVIDER` / resolve helpers."""
    normalized = (name or "").strip().lower()
    if normalized == "dummy":
        return DummyLLMProvider()
    if normalized == "ollama":
        return OllamaLLMProvider(settings=settings)
    if normalized == "anthropic":
        return AnthropicLLMProvider(settings=settings)
    if normalized == "openai":
        raise NotImplementedError(
            "LLM provider 'openai' is not implemented yet "
            "(optional after Anthropic). Use 'ollama', 'anthropic', or 'dummy'."
        )
    raise ValueError(
        f"Unknown LLM provider {normalized!r}. "
        f"Expected one of: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}."
    )
