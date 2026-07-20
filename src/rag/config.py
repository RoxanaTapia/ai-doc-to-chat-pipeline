"""Config and prompt loaders for the RAG generation stack."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# src/rag/config.py → repo root
APP_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"

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
