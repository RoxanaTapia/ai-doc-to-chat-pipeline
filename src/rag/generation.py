"""Public generate / stream entrypoints for grounded answers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

from rag.citations import INSUFFICIENT_CONTEXT_ANSWER
from rag.providers import get_llm_provider, resolve_llm_provider_name

# Verbatim sample-NDA lines used when DEMO_STUB_LLM is on (no API key).
_SAMPLE_TERM_LINE = (
    "This Agreement shall remain in effect for a period of three (3) years "
    "from the date of signing."
)
_SAMPLE_LAW_LINE = (
    "This Agreement shall be governed by the laws of Spain. "
    "Any dispute arising hereunder shall be resolved exclusively in the courts of Madrid."
)


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def stub_demo_answer(query: str) -> str | None:
    """Canned sample-NDA answers for captures. None when the env flag is off."""
    if not _env_flag("DEMO_STUB_LLM"):
        return None
    lowered = (query or "").lower()
    if "penalty" in lowered or "fee" in lowered:
        return INSUFFICIENT_CONTEXT_ANSWER
    if "law" in lowered or "dispute" in lowered:
        return f"Spain / Madrid\n\n{_SAMPLE_LAW_LINE}"
    if "long" in lowered or "last" in lowered or "term" in lowered:
        return f"3 years\n\n{_SAMPLE_TERM_LINE}"
    return INSUFFICIENT_CONTEXT_ANSWER


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")


def _normalize_untrusted_text(text: str, max_chars: int) -> str:
    """Trim and sanitize untrusted user/document text for prompting."""
    cleaned = CONTROL_CHARS_RE.sub("", text or "").strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars]
    return cleaned


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
        return INSUFFICIENT_CONTEXT_ANSWER
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
    canned = stub_demo_answer(safe_query)
    if canned is not None:
        return canned

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
    canned = stub_demo_answer(safe_query)
    if canned is not None:
        yield canned
        return

    provider_name = resolve_llm_provider_name(dummy_mode=dummy_mode)
    provider = get_llm_provider(provider_name)
    stream_fn = getattr(provider, "stream", None)
    if callable(stream_fn):
        yield from stream_fn(safe_context, safe_query)
        return
    answer = provider.generate(safe_context, safe_query)
    if answer:
        yield answer
