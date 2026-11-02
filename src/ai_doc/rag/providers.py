"""Swappable hosted LLM backends for grounded answer generation."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from ai_doc.rag.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_XAI_MODEL,
    PROMPTS_PATH,
    XAI_BASE_URL,
    load_generation_config,
    load_rag_prompt,
)

logger = logging.getLogger(__name__)

# `anthropic` 2.x defaults to API version 2026-10-21, which rejects `temperature`,
# `top_p`, and `top_k` on every model (400: "Extra inputs are not permitted").
# Haiku 4.5 still honors temperature on 2023-06-01, including 0 for do_sample=false.
_ANTHROPIC_SAMPLING_API_VERSION = "2023-06-01"

SUPPORTED_LLM_PROVIDERS = frozenset({"anthropic", "openai", "xai", "dummy"})
_PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
}
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
    """Placeholder text for UI / test dummy mode (no hosted LLM)."""
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


def _effective_temperature(settings: dict[str, Any]) -> float:
    """0.0 when sampling is off, so answers stay deterministic."""
    if settings["do_sample"]:
        return float(settings["temperature"])
    return 0.0


def _require_api_key(provider: str) -> str:
    """Return the selected provider's API key or raise a clear operator error."""
    env_name = _PROVIDER_API_KEYS[provider]
    key = (os.getenv(env_name) or "").strip()
    if not key:
        raise RuntimeError(
            f"{env_name} is not set. Add it to your local .env "
            f"(never commit the key). Required when LLM_PROVIDER={provider}."
        )
    return key


def _build_anthropic_llm(settings: dict[str, Any], api_key: str) -> ChatAnthropic:
    """Construct a ChatAnthropic client from generation settings.

    Haiku 4.5 accepts ``temperature`` on API version 2023-06-01 and rejects the
    field on 2026-10-21. Pin that header so a newer ``anthropic`` SDK cannot
    400 the request. Send temperature only (0.0 when ``do_sample`` is false)
    and never ``top_p`` (Haiku 4.5 rejects both together).
    """
    model = str(settings.get("anthropic_model") or DEFAULT_ANTHROPIC_MODEL)
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=_effective_temperature(settings),
        max_tokens=int(settings["max_new_tokens"]),
        timeout=float(settings["timeout_seconds"]),
        default_headers={"anthropic-version": _ANTHROPIC_SAMPLING_API_VERSION},
    )


def _build_openai_llm(settings: dict[str, Any], api_key: str) -> ChatOpenAI:
    """Construct a ChatOpenAI client. Temperature only, matching the other providers."""
    model = str(settings.get("openai_model") or DEFAULT_OPENAI_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=_effective_temperature(settings),
        max_tokens=int(settings["max_new_tokens"]),
        timeout=float(settings["timeout_seconds"]),
    )


def _build_xai_llm(settings: dict[str, Any], api_key: str) -> ChatOpenAI:
    """Construct a Grok client on the xAI OpenAI-compatible endpoint."""
    model = str(settings.get("xai_model") or DEFAULT_XAI_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=XAI_BASE_URL,
        temperature=_effective_temperature(settings),
        max_tokens=int(settings["max_new_tokens"]),
        timeout=float(settings["timeout_seconds"]),
    )


def _iter_llm_text_chunks(llm: Any, prompt: str) -> Iterator[str]:
    """Yield non-empty text pieces from `llm.stream(prompt)`."""
    for chunk in llm.stream(prompt):
        text = _content_to_text(chunk.content if hasattr(chunk, "content") else chunk)
        if text:
            yield text


def _complete_with(llm: Any, context: str, query: str) -> str:
    """Invoke `llm` with the RAG prompt and return stripped text."""
    prompt = _prompt_for_generation(context, query)
    response = llm.invoke(prompt)
    return _content_to_text(response.content if hasattr(response, "content") else response).strip()


def _settings_or_loaded(settings: dict[str, Any] | None) -> dict[str, Any]:
    if settings is None:
        return load_generation_config()
    return settings


def _generate_with_anthropic(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> str:
    """Generate answer text using Anthropic Claude (ChatAnthropic)."""
    api_key = _require_api_key("anthropic")
    resolved = _settings_or_loaded(settings)
    return _complete_with(_build_anthropic_llm(resolved, api_key), context, query)


def _generate_with_openai(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> str:
    """Generate answer text using OpenAI (ChatOpenAI)."""
    api_key = _require_api_key("openai")
    resolved = _settings_or_loaded(settings)
    return _complete_with(_build_openai_llm(resolved, api_key), context, query)


def _generate_with_xai(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> str:
    """Generate answer text using xAI Grok."""
    api_key = _require_api_key("xai")
    resolved = _settings_or_loaded(settings)
    return _complete_with(_build_xai_llm(resolved, api_key), context, query)


def _stream_with_anthropic(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer text chunks from Anthropic (`ChatAnthropic.stream`)."""
    api_key = _require_api_key("anthropic")
    resolved = _settings_or_loaded(settings)
    prompt = _prompt_for_generation(context, query)
    yield from _iter_llm_text_chunks(_build_anthropic_llm(resolved, api_key), prompt)


def _stream_with_openai(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer text chunks from OpenAI (`ChatOpenAI.stream`)."""
    api_key = _require_api_key("openai")
    resolved = _settings_or_loaded(settings)
    prompt = _prompt_for_generation(context, query)
    yield from _iter_llm_text_chunks(_build_openai_llm(resolved, api_key), prompt)


def _stream_with_xai(
    context: str,
    query: str,
    settings: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream answer text chunks from xAI Grok."""
    api_key = _require_api_key("xai")
    resolved = _settings_or_loaded(settings)
    prompt = _prompt_for_generation(context, query)
    yield from _iter_llm_text_chunks(_build_xai_llm(resolved, api_key), prompt)


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


class _HostedLLMProvider:
    """Hosted generator. A missing API key raises before any request is sent."""

    label = "Hosted"
    key_env = ""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings

    def _generate_fn(self, context: str, query: str, settings: dict[str, Any] | None) -> str:
        raise NotImplementedError

    def _stream_fn(
        self, context: str, query: str, settings: dict[str, Any] | None
    ) -> Iterator[str]:
        raise NotImplementedError

    def generate(self, context: str, query: str) -> str:
        settings = self._settings if self._settings is not None else load_generation_config()
        try:
            return self._generate_fn(context, query, settings)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and self.key_env in str(exc):
                raise
            logger.exception("%s generation failed: %s", self.label, exc)
            raise RuntimeError(f"{self.label} generation unavailable: {exc}") from exc

    def stream(self, context: str, query: str) -> Iterator[str]:
        settings = self._settings if self._settings is not None else load_generation_config()
        yield from _stream_with_generate_fallback(
            lambda: self._stream_fn(context, query, settings),
            lambda: self.generate(context, query),
            label=self.label,
        )


class AnthropicLLMProvider(_HostedLLMProvider):
    """Anthropic Claude generator (Haiku default)."""

    label = "Anthropic"
    key_env = "ANTHROPIC_API_KEY"

    def _generate_fn(self, context: str, query: str, settings: dict[str, Any] | None) -> str:
        return _generate_with_anthropic(context=context, query=query, settings=settings)

    def _stream_fn(
        self, context: str, query: str, settings: dict[str, Any] | None
    ) -> Iterator[str]:
        return _stream_with_anthropic(context=context, query=query, settings=settings)


class OpenAILLMProvider(_HostedLLMProvider):
    """OpenAI generator (GPT-4.1 mini default)."""

    label = "OpenAI"
    key_env = "OPENAI_API_KEY"

    def _generate_fn(self, context: str, query: str, settings: dict[str, Any] | None) -> str:
        return _generate_with_openai(context=context, query=query, settings=settings)

    def _stream_fn(
        self, context: str, query: str, settings: dict[str, Any] | None
    ) -> Iterator[str]:
        return _stream_with_openai(context=context, query=query, settings=settings)


class XAILLMProvider(_HostedLLMProvider):
    """xAI Grok generator."""

    label = "xAI"
    key_env = "XAI_API_KEY"

    def _generate_fn(self, context: str, query: str, settings: dict[str, Any] | None) -> str:
        return _generate_with_xai(context=context, query=query, settings=settings)

    def _stream_fn(
        self, context: str, query: str, settings: dict[str, Any] | None
    ) -> Iterator[str]:
        return _stream_with_xai(context=context, query=query, settings=settings)


def resolve_llm_provider_name(dummy_mode: bool = True) -> str:
    """Pick provider name from env, else map dummy mode to dummy vs the default.

    Resolution:
    1. `LLM_PROVIDER` env (non-empty) wins: anthropic, openai, xai, or dummy
    2. Else `dummy` when `dummy_mode` is set, otherwise `anthropic`
    """
    raw = os.getenv("LLM_PROVIDER")
    if raw is not None and (normalized := raw.strip()):
        return normalized.lower()
    if dummy_mode:
        return "dummy"
    return DEFAULT_LLM_PROVIDER


def get_llm_provider(
    name: str,
    settings: dict[str, Any] | None = None,
) -> LLMProvider:
    """Factory for generation backends selected by `LLM_PROVIDER` / resolve helpers."""
    normalized = (name or "").strip().lower()
    if normalized == "dummy":
        return DummyLLMProvider()
    factories = {
        "anthropic": AnthropicLLMProvider,
        "openai": OpenAILLMProvider,
        "xai": XAILLMProvider,
    }
    factory = factories.get(normalized)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider {normalized!r}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}."
        )
    return factory(settings=settings)
