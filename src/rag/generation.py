"""Public generate / stream entrypoints for grounded answers."""

from __future__ import annotations

import re
from collections.abc import Iterator

from rag.providers import get_llm_provider, resolve_llm_provider_name

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
