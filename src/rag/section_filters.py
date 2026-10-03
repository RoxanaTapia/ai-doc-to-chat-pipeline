"""Choose which chunks win when a question names a section."""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document

SECTION_IN_QUERY_RE = re.compile(r"section\s+(\d+)", re.IGNORECASE)
SECTION_SYMBOL_RE = re.compile(r"§\s*(\d+)")
# Walkthrough / eval Q1 without "Section 1": still route to the definition clause.
DEFINITION_QUERY_RE = re.compile(
    r"(?i)(?:"
    r"how is\b.{0,80}\bdefined"
    r"|what(?:'s| is)\s+confidential information\b"
    r"|definition of\s+confidential information"
    r")"
)
_OBLIGATION_HINT_RE = re.compile(
    r"(?i)\b(obligat|shall not|must not|duties|prohibit|requirements)\b"
)



def extract_target_section(query: str) -> str | None:
    """Parse target section from 'Section 3', '§3', or a definition-style Q1."""
    text = query or ""
    match = SECTION_IN_QUERY_RE.search(text)
    if match:
        return match.group(1)
    match = SECTION_SYMBOL_RE.search(text)
    if match:
        return match.group(1)
    if DEFINITION_QUERY_RE.search(text) and not _OBLIGATION_HINT_RE.search(text):
        return "1"
    return None


def _looks_like_section_one_definition(text: str) -> bool:
    """Treat a definition paragraph as section 1 when the question is about a meaning."""
    if not re.search(
        r"(?i)(?:confidential information.*(?:means|shall mean)|is defined as)",
        text or "",
    ):
        return False
    head = (text or "")[:400]
    return not re.search(
        r"(?i)\b(?:section\s+[2-9]|[2-9])\.\s+(?:exclusion|obligation|term|duration)",
        head,
    )


def chunk_contains_section(text: str, target_section: str) -> bool:
    """Content-aware: chunk text includes a header for the target section."""
    target = str(target_section)
    if re.search(rf"(?m)^\s*{re.escape(target)}\.\s", text or ""):
        return True
    if re.search(rf"(?i)\bsection\s+{re.escape(target)}\b", text or ""):
        return True
    target_slug = _slugify_title(target.replace("_", " "))
    for section_id, _start, _end, title in section_spans_in_chunk(text):
        if section_id == target or section_id == target_slug:
            return True
        if title and _slugify_title(title) == target_slug:
            return True
    return False


def chunk_content_matches_section(text: str, target_section: str) -> bool:
    """Whether chunk text belongs to the target section (header or section-1 definition)."""
    target = str(target_section)
    if chunk_contains_section(text, target):
        return True
    if target == "1" and _looks_like_section_one_definition(text):
        return True
    return False


def extract_section_text(text: str, target_section: str) -> str | None:
    """Return only the target section's text, trimming at the next numbered header."""
    target = str(target_section)
    spans = section_spans_in_chunk(text)
    for section_id, start, end, _title in spans:
        if section_id == target:
            return text[start:end].strip()
    pattern = rf"(?ms)^\s*{re.escape(target)}\.\s.+?(?=^\s*\d+\.\s|\Z)"
    match = re.search(pattern, text or "")
    if match:
        return match.group(0).strip()
    if target == "1" and _looks_like_section_one_definition(text):
        return text.strip()
    return None


def _doc_key(doc: Document) -> tuple:
    """Identify a chunk so the same text is not kept twice."""
    fingerprint = hashlib.blake2b(
        doc.page_content[:300].encode("utf-8", errors="ignore"),
        digest_size=8,
    ).digest()
    return (doc.metadata.get("page"), fingerprint)


def _section_matches(doc: Document, target: str) -> bool:
    """Metadata or content confirms the chunk belongs to the target section."""
    if chunk_content_matches_section(doc.page_content, target):
        return True
    return str(doc.metadata.get("section") or "") == target


def trim_document_to_section(doc: Document, target: str) -> Document | None:
    """Return a copy with page_content trimmed to the target section only."""
    if not chunk_content_matches_section(doc.page_content, target):
        return None
    trimmed = extract_section_text(doc.page_content, target)
    if not trimmed:
        return None
    metadata = dict(doc.metadata)
    metadata["section"] = target
    return Document(page_content=trimmed, metadata=metadata)


def apply_section_aware_retrieval(
    query: str,
    candidates: list[tuple[Document, float]],
    *,
    top_k: int,
    boost: float = 1.5,
    min_matching: int = 2,
) -> tuple[list[tuple[Document, float]], str | None]:
    """Prefer chunks from the named section when the question asks for one."""
    target = extract_target_section(query)
    if not target or not candidates:
        return candidates[:top_k], None

    boosted: list[tuple[Document, float]] = []
    for doc, score in candidates:
        if _section_matches(doc, target):
            boosted.append((doc, score * boost))
        else:
            boosted.append((doc, score))
    boosted.sort(key=lambda item: item[1], reverse=True)

    matching = [item for item in boosted if _section_matches(item[0], target)]
    unknown = [
        item
        for item in boosted
        if not item[0].metadata.get("section")
        and not chunk_content_matches_section(item[0].page_content, target)
    ]
    off_section = [item for item in boosted if item not in matching and item not in unknown]

    if len(matching) >= min_matching:
        pool = matching + unknown
    elif matching:
        pool = matching + unknown + off_section
    else:
        pool = boosted

    final: list[tuple[Document, float]] = []
    seen: set[tuple] = set()
    for doc, score in pool:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        final.append((doc, score))
        if len(final) >= top_k:
            break

    in_top = sum(1 for doc, _score in final if _section_matches(doc, target))
    warning = None
    if in_top < min_matching:
        warning = (
            f"Section {target}: only {in_top} tagged chunk(s) in top-{top_k}; "
            "showing best-effort matches."
        )
    return final[:top_k], warning


def apply_hard_section_context_filter(
    query: str,
    ranked: list[tuple[Document, float]],
    *,
    all_chunks: list[Document] | None,
    top_k: int,
    enabled: bool = True,
    min_chunks: int = 2,
) -> tuple[list[tuple[Document, float]], str | None]:
    """Keep only chunks from the named section so the model is not fed the wrong clause."""
    target = extract_target_section(query)
    if not enabled or not target or not ranked:
        return ranked[:top_k], None

    def _prepare(doc: Document, score: float) -> tuple[Document, float] | None:
        trimmed = trim_document_to_section(doc, target)
        if trimmed is None:
            return None
        return trimmed, score

    matching: list[tuple[Document, float]] = []
    for doc, score in ranked:
        prepared = _prepare(doc, score)
        if prepared:
            matching.append(prepared)

    if len(matching) < min_chunks and all_chunks:
        seen = {_doc_key(doc) for doc, _score in matching}
        for chunk in all_chunks:
            if not chunk_content_matches_section(chunk.page_content, target):
                continue
            trimmed = trim_document_to_section(chunk, target)
            if trimmed is None:
                continue
            key = _doc_key(trimmed)
            if key in seen:
                continue
            seen.add(key)
            matching.append((trimmed, 0.0))
            if len(matching) >= min_chunks:
                break

    if not matching:
        return ranked[:top_k], (
            f"Section {target}: no content-valid chunks — using unfiltered top-{top_k} for context."
        )

    if len(matching) < min_chunks:
        return matching[:top_k], (
            f"Section {target}: only {len(matching)} content-valid chunk(s) in context "
            f"(wanted ≥{min_chunks})."
        )

    return matching[:top_k], None


def chunk_on_section(doc: Document, target_section: str) -> bool:
    """Eval helper: content-aware section match for dev checklist."""
    target = str(target_section)
    if not chunk_content_matches_section(doc.page_content, target):
        return False
    trimmed = extract_section_text(doc.page_content, target)
    if trimmed and chunk_contains_section(doc.page_content, target):
        other_spans = [
            span
            for span in section_spans_in_chunk(doc.page_content)
            if span[0] != target and (span[2] - span[1]) > len(trimmed) * 0.5
        ]
        if other_spans:
            return False
    return True


def _text_looks_on_section(chunk_text: str, target_section: str) -> bool:
    """Check section membership from raw text."""
    return chunk_on_section(
        Document(page_content=chunk_text, metadata={}),
        target_section,
    )


# Loaded after chunking defines the span helpers, so the two modules can import each other.
from rag.chunking import _slugify_title, section_spans_in_chunk  # noqa: E402
