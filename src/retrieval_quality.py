"""Retrieval post-processing — tuning #3: dedupe and context sufficiency."""

from __future__ import annotations

import re

from langchain_core.documents import Document

INSUFFICIENT_CONTEXT_ANSWER = (
    "I could not find enough information in the document to answer."
)

OBLIGATION_QUERY_RE = re.compile(
    r"(?i)\b("
    r"must not|must agree|agree not|obligat|prohibit|prohibition|"
    r"duties|requirements|shall not|what must .+ not do"
    r")\b"
)
OBLIGATION_CONTEXT_MARKERS_RE = re.compile(
    r"(?i)\b("
    r"agree not to|shall not|refrain from|must not|prohibit|"
    r"shall:|restrict access|without .{0,40} (?:consent|authorization)"
    r")\b"
)


def _normalize_snippet(text: str, prefix_chars: int) -> str:
    return " ".join((text or "").split())[:prefix_chars].lower()


def _is_near_duplicate(snippet: str, seen: list[str]) -> bool:
    if not snippet:
        return True
    for prev in seen:
        if snippet == prev:
            return True
        if len(snippet) > 40 and len(prev) > 40 and (snippet in prev or prev in snippet):
            return True
    return False


def dedupe_similar_chunks(
    candidates: list[tuple[Document, float]],
    *,
    top_k: int,
    prefix_chars: int = 180,
    min_tiny_chars: int = 40,
) -> list[tuple[Document, float]]:
    """Drop near-duplicate chunks (same page, overlapping prefix, or tiny boilerplate)."""
    if not candidates:
        return []

    page_max_lens: dict[object, int] = {}
    for doc, _score in candidates:
        page = doc.metadata.get("page")
        length = len((doc.page_content or "").strip())
        page_max_lens[page] = max(page_max_lens.get(page, 0), length)

    final: list[tuple[Document, float]] = []
    seen_snippets: list[str] = []

    for doc, score in candidates:
        text = (doc.page_content or "").strip()
        snippet = _normalize_snippet(text, prefix_chars)
        if _is_near_duplicate(snippet, seen_snippets):
            continue

        page = doc.metadata.get("page")
        if (
            len(text) < min_tiny_chars
            and page_max_lens.get(page, 0) > len(text) * 2
        ):
            continue

        seen_snippets.append(snippet)
        final.append((doc, score))
        if len(final) >= top_k:
            break

    return final


def query_expects_obligations(query: str) -> bool:
    return bool(OBLIGATION_QUERY_RE.search(query or ""))


def context_has_obligation_markers(context: str) -> bool:
    return bool(OBLIGATION_CONTEXT_MARKERS_RE.search(context or ""))


def context_sufficient_for_query(query: str, context: str) -> tuple[bool, str | None]:
    """
    Return (True, None) when context is adequate to attempt generation.

    Tuning #3: obligation questions with no obligation language in context → insufficient.
    """
    if not (context or "").strip():
        return False, "empty_context"

    if query_expects_obligations(query) and not context_has_obligation_markers(context):
        return False, "obligation_markers_missing"

    return True, None
