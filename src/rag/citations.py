"""Citations, source ranking, and honesty guards."""

from __future__ import annotations

import re

from langchain_core.documents import Document

INSUFFICIENT_CONTEXT_ANSWER = (
    "This isn't in the document, so I won't guess. Ask about something it covers."
)
REFUSAL_CORE_PHRASE = "this isn't in the document"
_CURLY_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u02bc": "'",
    }
)
AMOUNT_QUERY_RE = re.compile(r"(?i)\b(fee|penalty|amount|liquidated damages)\b")
MONEY_AMOUNT_RE = re.compile(
    r"(?i)"
    r"([$€£]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|dollars?|euros?|pounds?))"
    r"|((?:usd|eur|gbp|dollars?|euros?|pounds?)\s?\d[\d,]*)"
)
_UNSPECIFIED_RE = re.compile(
    r"(?i)(\bnot (specified|stated|mentioned)\b|\bno\b.{0,60}\b(specified|stated|mentioned)\b)"
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
        if len(text) < min_tiny_chars and page_max_lens.get(page, 0) > len(text) * 2:
            continue

        seen_snippets.append(snippet)
        final.append((doc, score))
        if len(final) >= top_k:
            break

    return final


def normalize_refusal_text(text: str) -> str:
    """Lowercase, straighten curly quotes, and collapse whitespace."""
    return " ".join((text or "").translate(_CURLY_QUOTES).lower().split())


def query_expects_amount(query: str) -> bool:
    """True when the question asks for a fee, penalty, or amount."""
    return bool(AMOUNT_QUERY_RE.search(query or ""))


def _has_substantive_content_besides_unspecified(normalized: str) -> bool:
    """True when text still has a fact after the 'not specified' clause is removed."""
    remainder = _UNSPECIFIED_RE.sub(" ", normalized)
    tokens = [tok for tok in _TOKEN_RE.findall(remainder) if tok not in _OVERLAP_STOPWORDS]
    return any(tok.isdigit() for tok in tokens) or len(tokens) >= 2


def is_insufficient_answer(text: str, query: str | None = None) -> bool:
    """True for the sentinel, or a 'not specified' miss on a fee question or empty reply."""
    normalized = normalize_refusal_text(text)
    if not normalized:
        return False
    if REFUSAL_CORE_PHRASE in normalized:
        return True
    if not _UNSPECIFIED_RE.search(normalized):
        return False
    if query_expects_amount(query or ""):
        return True
    return not _has_substantive_content_besides_unspecified(normalized)


def context_has_money_amount(context: str) -> bool:
    """True when retrieved text names a currency amount (not a bare year or count)."""
    return bool(MONEY_AMOUNT_RE.search(context or ""))


def visitor_refusal_answer(text: str, query: str | None = None) -> str:
    """Show the exact sentinel whenever the raw answer is a miss."""
    if is_insufficient_answer(text, query):
        return INSUFFICIENT_CONTEXT_ANSWER
    return text or ""


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

    if query_expects_amount(query) and not context_has_money_amount(context):
        return False, "amount_missing"

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
    return {tok for tok in _TOKEN_RE.findall((text or "").lower()) if tok not in _OVERLAP_STOPWORDS}


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


def _answer_digit_tokens(answer: str) -> set[str]:
    """Digit tokens in the answer (e.g. ``3`` from '3 years')."""
    return {tok for tok in _content_tokens(answer) if tok.isdigit()}


def _chunk_contains_answer_digits(chunk_text: str, digits: set[str]) -> bool:
    """True when every answer digit appears in the chunk text."""
    if not digits:
        return True
    text = chunk_text or ""
    return all(digit in text for digit in digits)


def _chunk_supports_answer_tokens(chunk_text: str, answer: str) -> bool:
    """True when the chunk carries the answer's numbers and at least one other token."""
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return False
    chunk_tokens = _content_tokens(chunk_text)
    digits = {tok for tok in answer_tokens if tok.isdigit()}
    words = answer_tokens - digits
    if digits and not _chunk_contains_answer_digits(chunk_text, digits):
        return False
    if words:
        return bool(words & chunk_tokens)
    return bool(digits) and digits <= chunk_tokens


def filter_docs_overlapping_answer(
    docs: list[Document],
    answer: str,
    *,
    min_score: float = DEFAULT_ANSWER_OVERLAP_MIN_SCORE,
) -> list[Document]:
    """
    Keep Sources chunks that support ``answer``, in the given retrieval order.

    Prefer Jaccard overlap at ``min_score``. Short answers like "3 years" often
    miss that floor, so fall back to chunks that contain the answer's numbers
    and at least one other content token. If nothing supports the answer, keep
    only the first retrieved chunk, not the whole unfiltered list.
    """
    if not docs or not (answer or "").strip():
        return docs

    digits = _answer_digit_tokens(answer)
    scored_hits = [
        doc for doc in docs if token_overlap_score(answer, doc.page_content or "") >= min_score
    ]
    if digits:
        numbered = [
            doc
            for doc in scored_hits
            if _chunk_contains_answer_digits(doc.page_content or "", digits)
        ]
        scored_hits = numbered
    if scored_hits:
        return scored_hits

    token_hits = [
        doc for doc in docs if _chunk_supports_answer_tokens(doc.page_content or "", answer)
    ]
    if token_hits:
        return token_hits
    return docs[:1]


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

    if is_insufficient_answer(answer or "", query):
        return []

    target_section = extract_target_section(query) if query else None
    # Keep retrieval/ranked order. Do not re-sort by page or raw similarity.
    ordered = list(retrieved_docs)
    if answer:
        ordered = filter_docs_overlapping_answer(ordered, answer)
    if target_section:
        on_section = [doc for doc in ordered if chunk_on_section(doc, target_section)]
        if on_section:
            ordered = on_section
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
