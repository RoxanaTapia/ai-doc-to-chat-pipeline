import logging
import os
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
APP_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"
SUPPORTED_LLM_PROVIDERS = frozenset({"ollama", "anthropic", "openai", "dummy"})
# Current Anthropic Haiku (fast demo/video). Override with ANTHROPIC_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROMPT_TEMPLATE = (
    "You are a precise and concise legal assistant.\n"
    "Answer ONLY using information from the provided context.\n"
    "If there is no relevant information, say: "
    '"I could not find enough information in the document to answer."\n\n'
    "Extracted context:\n"
    "{context}\n\n"
    "User question:\n"
    "{question}\n\n"
    "Answer (keep it short, clear, and cite the source when possible):"
)
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


def _normalize_untrusted_text(text: str, max_chars: int) -> str:
    """Trim and sanitize untrusted user/document text for prompting."""
    cleaned = CONTROL_CHARS_RE.sub("", text or "").strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars]
    return cleaned


@lru_cache(maxsize=1)
def load_rag_prompt() -> str:
    """Load and cache the raw RAG prompt template from YAML config."""
    if not PROMPTS_PATH.exists():
        return DEFAULT_PROMPT_TEMPLATE
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    prompt_cfg = config.get("rag_prompt", {}) or {}
    direct_template = prompt_cfg.get("template")
    if direct_template:
        return direct_template
    system_template = (prompt_cfg.get("system_template") or "").strip()
    user_template = (prompt_cfg.get("user_template") or "").strip()
    if system_template and user_template:
        return f"{system_template}\n\n{user_template}"
    return DEFAULT_PROMPT_TEMPLATE


@lru_cache(maxsize=1)
def load_generation_config() -> dict[str, Any]:
    """Load generation config with optional env overrides."""
    config = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    generation = (config.get("rag", {}) or {}).get("generation", {}) or {}

    def _env_text(name: str) -> str | None:
        """Return stripped env var text, treating empty values as unset."""
        raw = os.getenv(name)
        if raw is None:
            return None
        normalized = raw.strip()
        return normalized if normalized else None

    def _env_bool(name: str, default: bool) -> bool:
        raw = _env_text(name)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def _env_float(name: str, default: float) -> float:
        raw = _env_text(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
            return float(default)

    def _env_int(name: str, default: int) -> int:
        raw = _env_text(name)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
            return int(default)

    default_model = str(generation.get("model", "llama3.1:8b"))
    model_override = _env_text("OLLAMA_MODEL")
    # Env wins when set; YAML `provider` is informational default for operators.
    yaml_provider = str(generation.get("provider", "") or "").strip().lower()
    env_provider = _env_text("LLM_PROVIDER")
    anthropic_model = _env_text("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL

    return {
        "provider": (env_provider or yaml_provider or "").lower() or None,
        "model": model_override or default_model,
        "anthropic_model": anthropic_model,
        "temperature": _env_float("OLLAMA_TEMPERATURE", float(generation.get("temperature", 0.3))),
        "top_p": _env_float("OLLAMA_TOP_P", float(generation.get("top_p", 0.9))),
        "max_new_tokens": _env_int(
            "OLLAMA_MAX_NEW_TOKENS",
            int(generation.get("max_new_tokens", 256)),
        ),
        "do_sample": _env_bool("OLLAMA_DO_SAMPLE", bool(generation.get("do_sample", False))),
        "fallback_to_dummy_on_error": _env_bool(
            "OLLAMA_FALLBACK_TO_DUMMY",
            bool(generation.get("fallback_to_dummy_on_error", False)),
        ),
        "num_ctx": _env_int("OLLAMA_NUM_CTX", int(generation.get("num_ctx", 4096))),
        "timeout_seconds": _env_int(
            "OLLAMA_TIMEOUT_SECONDS",
            int(generation.get("timeout_seconds", 90)),
        ),
    }


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
    """Construct a ChatAnthropic client from generation settings."""
    effective_temperature = settings["temperature"] if settings["do_sample"] else 0.0
    model = str(settings.get("anthropic_model") or DEFAULT_ANTHROPIC_MODEL)
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=effective_temperature,
        max_tokens=int(settings["max_new_tokens"]),
        top_p=settings["top_p"],
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


def _prepare_generation_inputs(
    context: str,
    query: str,
) -> tuple[str, str] | str:
    """Validate/sanitize inputs. Return `(safe_context, safe_query)` or an error message."""
    safe_query = _normalize_untrusted_text(query, max_chars=2000)
    if not safe_query:
        return "Please enter a non-empty question."
    safe_context = _normalize_untrusted_text(context, max_chars=12000)
    if not safe_context:
        return "I could not find relevant information in the document to answer this question."
    return safe_context, safe_query


def generate_answer(context: str, query: str, dummy_mode: bool = True) -> str:
    """Generate a response from retrieved context and user question.

    Provider selection: explicit `LLM_PROVIDER` env wins; when unset, `dummy_mode`
    (and thus Streamlit's `USE_DUMMY_GENERATOR`) maps to dummy vs ollama.
    """
    prepared = _prepare_generation_inputs(context, query)
    if isinstance(prepared, str):
        return prepared
    safe_context, safe_query = prepared

    provider_name = resolve_llm_provider_name(dummy_mode=dummy_mode)
    provider = get_llm_provider(provider_name)
    return provider.generate(safe_context, safe_query)


def generate_answer_stream(
    context: str,
    query: str,
    dummy_mode: bool = True,
) -> Iterator[str]:
    """Yield answer text chunks for progressive UI (`st.write_stream`).

    Same provider selection and input validation as `generate_answer`. Providers
    that support native streaming use `.stream()`; if streaming fails before any
    chunk, the full `generate()` result is yielded as one chunk.
    """
    prepared = _prepare_generation_inputs(context, query)
    if isinstance(prepared, str):
        yield prepared
        return
    safe_context, safe_query = prepared

    provider_name = resolve_llm_provider_name(dummy_mode=dummy_mode)
    provider = get_llm_provider(provider_name)
    stream_fn = getattr(provider, "stream", None)
    if callable(stream_fn):
        yield from stream_fn(safe_context, safe_query)
        return
    answer = provider.generate(safe_context, safe_query)
    if answer:
        yield answer
