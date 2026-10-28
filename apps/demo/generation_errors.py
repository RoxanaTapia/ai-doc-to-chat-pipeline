"""Visitor-facing copy for generation failures.

Provider exceptions stay in the log. Chat and the error banner never repeat
the raw API body.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def anthropic_user_reason(exc: BaseException) -> str:
    """Map an Anthropic failure to a short sentence safe to show in the demo."""
    logger.warning("Anthropic answer failed: %s", exc)
    error_text = str(exc).lower()
    if "api_key" in error_text or "authentication" in error_text or "401" in error_text:
        return "Anthropic rejected the API key. Check `ANTHROPIC_API_KEY` in `.env`."
    if "rate" in error_text or "429" in error_text:
        return "Anthropic rate limit reached. Wait a moment, then retry."
    if "timeout" in error_text or "timed out" in error_text:
        return "Anthropic timed out. Retry with a shorter question."
    if "temperature" in error_text or "top_p" in error_text or "top_k" in error_text:
        return "The answer service could not use these settings. Please try again."
    return "Could not generate an Anthropic answer right now."
