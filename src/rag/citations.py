"""Citations, source ranking, and honesty guards."""

from __future__ import annotations

import re

from langchain_core.documents import Document

INSUFFICIENT_CONTEXT_ANSWER = (
    "I could not find enough information in the document to answer."
)

# Light English stopwords for answer↔chunk overlap (keep the set tiny).
_OVERLAP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Default Jaccard floor for Sources answer-overlap (#82). Modest because
# answers are short relative to chunks; tune upward if citations stay noisy.
DEFAULT_ANSWER_OVERLAP_MIN_SCORE = 0.08

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


def _similarity_score(doc: Document) -> float:
    """UI similarity from metadata; missing/non-numeric → lowest."""
    score = doc.metadata.get("similarity")
    if isinstance(score, (int, float)):
        return float(score)
    return float("-inf")


def sort_source_docs(
    docs: list[Document],
    *,
    target_section: str | None,
) -> list[Document]:
    """
    Order Sources for display: on-section first, then similarity descending.

    When ``target_section`` is set, section-matched chunks rank above off-section
    ones; within each group, higher ``metadata["similarity"]`` wins. With no
    target section, order by similarity only. Missing scores sort last.
    """
    if not docs:
        return []

    if not target_section:
        return sorted(docs, key=_similarity_score, reverse=True)

    from rag.chunking import chunk_on_section

    return sorted(
        docs,
        key=lambda doc: (chunk_on_section(doc, target_section), _similarity_score(doc)),
        reverse=True,
    )


def _content_tokens(text: str) -> set[str]:
    """Lowercase alnum tokens minus a light stopword list."""
    return {
        tok
        for tok in _TOKEN_RE.findall((text or "").lower())
        if tok not in _OVERLAP_STOPWORDS
    }


def token_overlap_score(answer: str, chunk_text: str) -> float:
    """
    Deterministic token Jaccard between answer and chunk (0.0–1.0).

    Empty answer or empty chunk → 0.0. No LLM calls.
    """
    answer_tokens = _content_tokens(answer)
    chunk_tokens = _content_tokens(chunk_text)
    if not answer_tokens or not chunk_tokens:
        return 0.0

    intersection = answer_tokens & chunk_tokens
    union = answer_tokens | chunk_tokens
    return len(intersection) / len(union)


def filter_docs_overlapping_answer(
    docs: list[Document],
    answer: str,
    *,
    min_score: float = DEFAULT_ANSWER_OVERLAP_MIN_SCORE,
) -> list[Document]:
    """
    Keep Sources chunks whose token overlap with ``answer`` is >= ``min_score``.

    Safe fallback: if the answer is blank, or no doc clears the threshold,
    return the original ``docs`` list unchanged (same order / identity).
    """
    if not docs or not (answer or "").strip():
        return docs

    kept = [
        doc
        for doc in docs
        if token_overlap_score(answer, doc.page_content or "") >= min_score
    ]
    return kept if kept else docs


def readable_excerpt(text: str, max_chars: int) -> str:
    """Trim chunk text at word or sentence boundaries for source previews."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    snippet = cleaned[:max_chars]
    for sep in (". ", "; ", ", ", " "):
        cut = snippet.rfind(sep)
        if cut > max_chars // 2:
            return f"{snippet[: cut + len(sep)].strip()}…"
    return f"{snippet.rstrip()}…"


def assemble_context(
    raw_results: list[tuple[Document, float]],
    *,
    use_page_separators: bool,
) -> str:
    """Build the exact context string that is fed to the generator."""
    if use_page_separators:
        chunks_with_pages = []
        for doc, _score in raw_results:
            page = doc.metadata.get("page", "?")
            chunks_with_pages.append(f"─── Page {page} ───\nPage {page}: {doc.page_content}")
        return "\n".join(chunks_with_pages)
    return "\n\n".join(doc.page_content for doc, _score in raw_results)


def build_sources_payload(
    retrieved_docs: list[Document],
    *,
    query: str | None = None,
    answer: str | None = None,
    display_max: int,
    preview_chars: int,
    checklist_preview_chars: int = 80,
) -> list[dict]:
    """Create a compact serializable source payload per assistant answer."""
    from rag.chunking import chunk_on_section, extract_target_section

    target_section = extract_target_section(query) if query else None
    ordered = sort_source_docs(retrieved_docs, target_section=target_section)
    if answer:
        ordered = filter_docs_overlapping_answer(ordered, answer)
    payload = []
    for chunk in ordered[:display_max]:
        similarity = chunk.metadata.get("similarity")
        entry = {
            "page": chunk.metadata.get("page", "N/A"),
            "score": round(similarity, 3) if isinstance(similarity, (float, int)) else "N/A",
            "preview": readable_excerpt(chunk.page_content, preview_chars),
        }
        if target_section is not None:
            entry["checklist_preview"] = readable_excerpt(
                chunk.page_content,
                checklist_preview_chars,
            )
            entry["on_section"] = chunk_on_section(chunk, target_section)
        payload.append(entry)
    return payload
