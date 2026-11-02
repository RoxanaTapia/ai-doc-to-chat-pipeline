"""Visitor-facing copy for generation failures.

Provider exceptions stay in the log. Chat and the error banner never repeat
the raw API body.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def hosted_user_reason(exc: BaseException, *, label: str, key_env: str) -> str:
    """Map a hosted-model failure to a short sentence safe to show in the demo."""
    logger.warning("%s answer failed: %s", label, exc)
    error_text = str(exc).lower()
    if (
        "api_key" in error_text
        or key_env.lower() in error_text
        or "authentication" in error_text
        or "401" in error_text
    ):
        return f"{label} rejected the API key. Check `{key_env}` in `.env`."
    if "rate" in error_text or "429" in error_text:
        return f"{label} rate limit reached. Wait a moment, then retry."
    if "timeout" in error_text or "timed out" in error_text:
        return f"{label} timed out. Retry with a shorter question."
    if "temperature" in error_text or "top_p" in error_text or "top_k" in error_text:
        return "The answer service could not use these settings. Please try again."
    return f"Could not generate an {label} answer right now."


def anthropic_user_reason(exc: BaseException) -> str:
    """Map an Anthropic failure to a short sentence safe to show in the demo."""
    return hosted_user_reason(exc, label="Anthropic", key_env="ANTHROPIC_API_KEY")
