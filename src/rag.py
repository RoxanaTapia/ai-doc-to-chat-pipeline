import os
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
APP_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"
DEFAULT_PROMPT_TEMPLATE = (
    "You are a precise and concise legal assistant.\n"
    "Answer ONLY using information from the provided context.\n"
    'If there is no relevant information, say: "I could not find enough information in the document to answer."\n\n'
    "Extracted context:\n"
    "{context}\n\n"
    "User question:\n"
    "{question}\n\n"
    "Answer (keep it short, clear, and cite the source when possible):"
)


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

    default_model = str(generation.get("model", "phi3:mini"))
    model_override = _env_text("OLLAMA_MODEL")

    return {
        "model": model_override or default_model,
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


def _generate_with_ollama(context: str, query: str, settings: dict[str, Any] | None = None) -> str:
    """Generate answer text using local Ollama runtime."""
    if settings is None:
        settings = load_generation_config()
    prompt_template = load_rag_prompt() if PROMPTS_PATH.exists() else DEFAULT_PROMPT_TEMPLATE
    prompt = _format_prompt(prompt_template, context=context, query=query)
    effective_temperature = settings["temperature"] if settings["do_sample"] else 0.0

    llm = ChatOllama(
        model=settings["model"],
        temperature=effective_temperature,
        top_p=settings["top_p"],
        num_predict=settings["max_new_tokens"],
        num_ctx=settings["num_ctx"],
        timeout=settings["timeout_seconds"],
    )
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, "content") else str(response).strip()


def generate_answer(context: str, query: str, dummy_mode: bool = True) -> str:
    """Generate a response from retrieved context and user question."""
    safe_query = _normalize_untrusted_text(query, max_chars=2000)
    if not safe_query:
        return "Please enter a non-empty question."
    safe_context = _normalize_untrusted_text(context, max_chars=12000)
    if not safe_context:
        return "I could not find relevant information in the document to answer this question."

    def _dummy_response() -> str:
        context_preview = safe_context[:400]
        truncation_suffix = "..." if len(safe_context) > 400 else ""
        return (
            "Dummy answer:\n"
            f"{safe_query}\n\n"
            "Based on the document:\n"
            f"{context_preview}{truncation_suffix}"
        )

    if dummy_mode:
        return _dummy_response()

    settings = load_generation_config()
    try:
        return _generate_with_ollama(context=safe_context, query=safe_query, settings=settings)
    except Exception as exc:
        logger.exception("Ollama generation failed: %s", exc)
        if settings.get("fallback_to_dummy_on_error", False):
            return (
                "Ollama unavailable, falling back to dummy mode.\n\n"
                f"{_dummy_response()}"
            )
        raise RuntimeError(f"Ollama generation unavailable: {exc}") from exc
