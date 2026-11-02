"""Config and prompt loaders for the RAG generation stack."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# src/ai_doc/rag/config.py → repo root
APP_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"

# Fast hosted defaults. Override with ANTHROPIC_MODEL, OPENAI_MODEL, or XAI_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_XAI_MODEL = "grok-4.7"
# xAI's OpenAI-compatible chat endpoint. The key is XAI_API_KEY, not OPENAI_API_KEY.
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_PROMPT_TEMPLATE = (
    "You are a precise and concise legal assistant.\n"
    "Answer ONLY using information from the provided context.\n"
    "If the context does not contain the answer, reply with exactly this "
    "sentence and nothing else: "
    "\"This isn't in the document, so I won't guess. Ask about something it covers.\"\n"
    "Do not give partial or adjacent answers. If a penalty, fee, or amount is "
    "not in the context, do not infer from nearby remedies or injunctive relief.\n\n"
    "Extracted context:\n"
    "{context}\n\n"
    "User question:\n"
    "{question}\n\n"
    "Answer (keep it short, clear, and cite the source when possible):"
)


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

    # Env wins when set. YAML `provider` documents the hosted default.
    # Runtime selection still lives in resolve_llm_provider_name (dummy mode
    # applies only when LLM_PROVIDER is empty).
    yaml_provider = str(generation.get("provider", "") or "").strip().lower()
    env_provider = _env_text("LLM_PROVIDER")

    def _model(env_name: str, yaml_key: str, default: str) -> str:
        return _env_text(env_name) or str(generation.get(yaml_key) or default)

    return {
        "provider": (env_provider or yaml_provider or DEFAULT_LLM_PROVIDER).lower(),
        "anthropic_model": _model("ANTHROPIC_MODEL", "anthropic_model", DEFAULT_ANTHROPIC_MODEL),
        "openai_model": _model("OPENAI_MODEL", "openai_model", DEFAULT_OPENAI_MODEL),
        "xai_model": _model("XAI_MODEL", "xai_model", DEFAULT_XAI_MODEL),
        "temperature": _env_float("LLM_TEMPERATURE", float(generation.get("temperature", 0.3))),
        "top_p": _env_float("LLM_TOP_P", float(generation.get("top_p", 0.9))),
        "max_new_tokens": _env_int(
            "LLM_MAX_NEW_TOKENS",
            int(generation.get("max_new_tokens", 256)),
        ),
        "do_sample": _env_bool("LLM_DO_SAMPLE", bool(generation.get("do_sample", False))),
        "timeout_seconds": _env_int(
            "LLM_TIMEOUT_SECONDS",
            int(generation.get("timeout_seconds", 90)),
        ),
    }
