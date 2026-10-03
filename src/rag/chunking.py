"""Split PDF pages into chunks. Section-filter names stay importable from here."""

from __future__ import annotations

import re

from langchain_core.documents import Document

NUMBERED_CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+(.+)$")
TITLE_LINE_HEADER_RE = re.compile(
    r"(?m)^\s*([A-Z][A-Za-z][A-Za-z0-9 /&'-]{0,48})\.\s*(?:\n|$)"
)
# Line-anchored: mid-sentence "defined in Section 1" must not become a header.
SECTION_LABEL_RE = re.compile(r"(?im)^\s*section\s+(\d+)\b")

_SKIP_TITLE_PREFIXES = (
    "Note to",
    "I ",
    "The ",
    "If ",
    "Whereas",
    "Place and",
    "Signature",
)

# TOC leaders ("Title ........ 3") and FAQ-style numbered questions are not clause headers.
_TOC_LEADER_RE = re.compile(r"\.{2,}\s*\d+\s*$")
_INTERROGATIVE_START_RE = re.compile(
    r"(?i)^(?:how|what|why|when|where|who|whom|whose|which|"
    r"does|do|did|is|are|was|were|can|could|should|will|would|may)\b"
)
_NOTE_FOR_RE = re.compile(r"(?i)^note for\b")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\.\s+")


def _slugify_title(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "section"


def _valid_title_header(title: str) -> bool:
    cleaned = (title or "").strip()
    if len(cleaned) < 3:
        return False
    if any(cleaned.startswith(prefix) for prefix in _SKIP_TITLE_PREFIXES):
        return False
    if cleaned.isupper() and len(cleaned.split()) >= 3:
        return False
    words = cleaned.split()
    if len(words) >= 2:
        allowed_lower = {"of", "and", "or", "the", "a", "an"}
        significant = [word for word in words if word.lower() not in allowed_lower]
        if significant and not all(word[0].isupper() for word in significant):
            return False
    return True


def _looks_like_toc_entry(text: str) -> bool:
    """True for TOC rows with leader dots and a trailing page number."""
    return bool(_TOC_LEADER_RE.search((text or "").strip()))


def _looks_like_faq_question(text: str) -> bool:
    """True for eval/FAQ lines like 'How is X defined…?' or numbered questions."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    body = _NUMBERED_LINE_RE.sub("", cleaned, count=1).strip()
    if not body:
        return False
    if body.endswith("?"):
        return True
    return bool(_INTERROGATIVE_START_RE.match(body))


def _valid_numbered_header(title: str) -> bool:
    """Reject TOC leaders and FAQ-style fragments as numbered clause headers."""
    cleaned = (title or "").strip()
    if not cleaned:
        return False
    if _looks_like_toc_entry(cleaned):
        return False
    if _looks_like_faq_question(cleaned):
        return False
    return True


def _is_bleed_line(line: str) -> bool:
    """Lines that should not stick to a section body (TOC rows, FAQ prompts, note banners)."""
    stripped = (line or "").strip()
    if not stripped:
        return False
    if _NOTE_FOR_RE.match(stripped):
        return True
    if _looks_like_toc_entry(stripped):
        return True
    if _NUMBERED_LINE_RE.match(stripped) and _looks_like_faq_question(stripped):
        return True
    return False


def _peel_trailing_bleed(text: str) -> tuple[str, str]:
    """
    Peel trailing TOC/FAQ/note lines from a section body.

    Returns (clean_body, bleed_text). Bleed is empty when nothing was peeled.
    """
    lines = (text or "").splitlines()
    if not lines:
        return "", ""

    bleed_start = len(lines)
    index = len(lines) - 1
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            index -= 1
            continue
        if _is_bleed_line(lines[index]):
            bleed_start = index
            index -= 1
            continue
        break

    while bleed_start > 0 and not lines[bleed_start - 1].strip():
        bleed_start -= 1

    body = "\n".join(lines[:bleed_start]).strip()
    bleed = "\n".join(lines[bleed_start:]).strip()
    return body, bleed


def find_legal_headers(text: str) -> list[tuple[int, str, str]]:
    """Return sorted (char_offset, section_id, title) for numbered and title-style headers."""
    headers: list[tuple[int, str, str]] = []
    seen_at: set[int] = set()

    for match in NUMBERED_CLAUSE_RE.finditer(text or ""):
        title = match.group(2).strip()
        if not _valid_numbered_header(title):
            continue
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, match.group(1), title[:80]))

    for match in SECTION_LABEL_RE.finditer(text or ""):
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, match.group(1), ""))

    for match in TITLE_LINE_HEADER_RE.finditer(text or ""):
        title = match.group(1).strip()
        if not _valid_title_header(title):
            continue
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, _slugify_title(title), title))

    headers.sort(key=lambda item: item[0])
    return headers


def find_section_headers(text: str) -> list[tuple[int, str, str]]:
    """Return sorted (char_offset, section_id, title_snippet) headers in page text."""
    return find_legal_headers(text)


def split_documents_by_legal_headers(page_docs: list[Document]) -> list[Document]:
    """
    Split each page into header-bounded sections before character chunking (tuning #2).

    Keeps preamble (text before the first header) as its own document when non-empty.
    Peels trailing TOC/FAQ bleed from section bodies into a separate notes document.
    """
    section_docs: list[Document] = []
    for page_doc in page_docs:
        text = page_doc.page_content or ""
        base_metadata = dict(page_doc.metadata)
        headers = find_legal_headers(text)
        page_bleed: list[str] = []

        if not headers:
            section_docs.append(page_doc)
            continue

        if headers[0][0] > 0:
            preamble = text[: headers[0][0]].strip()
            if preamble:
                section_docs.append(
                    Document(
                        page_content=preamble,
                        metadata={**base_metadata, "section_title": "preamble"},
                    )
                )

        for index, (pos, section_id, title) in enumerate(headers):
            end = headers[index + 1][0] if index + 1 < len(headers) else len(text)
            section_text = text[pos:end].strip()
            if not section_text:
                continue
            body, bleed = _peel_trailing_bleed(section_text)
            if bleed:
                page_bleed.append(bleed)
            if not body:
                continue
            section_docs.append(
                Document(
                    page_content=body,
                    metadata={
                        **base_metadata,
                        "section": section_id,
                        "section_title": title,
                    },
                )
            )

        if page_bleed:
            notes = "\n\n".join(page_bleed).strip()
            if notes:
                section_docs.append(
                    Document(
                        page_content=notes,
                        metadata={**base_metadata, "section_title": "notes"},
                    )
                )

    return section_docs if section_docs else page_docs


def section_at_offset(headers: list[tuple[int, str, str]], offset: int) -> tuple[str | None, str | None]:
    """Section active at character offset (nearest header at or above offset)."""
    section: str | None = None
    title: str | None = None
    for start, section_id, section_title in headers:
        if start <= offset:
            section = section_id
            title = section_title or None
        else:
            break
    return section, title


def section_spans_in_chunk(text: str) -> list[tuple[str, int, int, str | None]]:
    """Return (section_id, start, end, title_snippet) spans for headers inside chunk text."""
    headers: list[tuple[int, str, str | None]] = []
    seen_at: set[int] = set()

    for match in NUMBERED_CLAUSE_RE.finditer(text or ""):
        title = match.group(2).strip()
        if not _valid_numbered_header(title):
            continue
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, match.group(1), title[:80]))

    for match in TITLE_LINE_HEADER_RE.finditer(text or ""):
        title = match.group(1).strip()
        if not _valid_title_header(title):
            continue
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, _slugify_title(title), title))

    if not headers:
        return []
    headers.sort(key=lambda item: item[0])
    spans: list[tuple[str, int, int, str | None]] = []
    for index, (pos, section_id, title) in enumerate(headers):
        end = headers[index + 1][0] if index + 1 < len(headers) else len(text)
        spans.append((section_id, pos, end, title))
    return spans


def dominant_section_in_chunk(text: str) -> tuple[str | None, str | None]:
    """Section with the largest character span inside the chunk (R4-B.1)."""
    spans = section_spans_in_chunk(text)
    if not spans:
        return None, None
    section_id, _start, _end, title = max(spans, key=lambda item: item[2] - item[1])
    return section_id, title


def _section_from_chunk_start(text: str) -> tuple[str | None, str | None]:
    """If the chunk opens with a numbered clause, treat that as its section."""
    match = re.match(r"^\s*(\d+)\.\s+([^\n]+)", text or "")
    if not match:
        return None, None
    return match.group(1), match.group(2).strip()[:80]


def annotate_chunk_sections(
    chunks: list[Document],
    page_texts: dict[int, str],
) -> list[Document]:
    """R4-B: attach metadata.section from in-chunk headers (dominant span) + page fallback."""
    header_cache: dict[int, list[tuple[int, str, str]]] = {}
    for chunk in chunks:
        page = chunk.metadata.get("page")
        page_text = page_texts.get(page, "") if isinstance(page, int) else ""
        if isinstance(page, int) and page not in header_cache:
            header_cache[page] = find_legal_headers(page_text)

        text = chunk.page_content or ""
        pre_section = chunk.metadata.get("section")
        pre_title = chunk.metadata.get("section_title")

        section, title = dominant_section_in_chunk(text)

        if not section and pre_section:
            section, title = str(pre_section), pre_title if isinstance(pre_title, str) else None

        if not section:
            headers = header_cache.get(page, []) if isinstance(page, int) else []
            start_index = int(chunk.metadata.get("start_index", 0) or 0)
            section, title = section_at_offset(headers, start_index)

        if not section:
            section, title = _section_from_chunk_start(text)

        spans = section_spans_in_chunk(text)
        if len(spans) > 1:
            chunk.metadata["section_span"] = "multi"

        if section:
            chunk.metadata["section"] = section
        if title:
            chunk.metadata["section_title"] = title
    return chunks


def limit_chunks_per_page(
    candidates: list[tuple[Document, float]],
    *,
    top_k: int,
    max_per_page: int = 2,
) -> list[tuple[Document, float]]:
    """R4-D: cap how many chunks from the same page appear in top-k."""
    if max_per_page <= 0:
        return candidates[:top_k]

    page_counts: dict[object, int] = {}
    limited: list[tuple[Document, float]] = []
    for doc, score in candidates:
        page = doc.metadata.get("page", "?")
        if page_counts.get(page, 0) >= max_per_page:
            continue
        page_counts[page] = page_counts.get(page, 0) + 1
        limited.append((doc, score))
        if len(limited) >= top_k:
            break
    return limited


def chunk_pages(
    page_docs: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    split_on_legal_headers: bool,
) -> list[Document]:
    """Split page documents at headers, then into overlapping windows."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    docs_to_split = (
        split_documents_by_legal_headers(page_docs)
        if split_on_legal_headers
        else page_docs
    )
    chunks = splitter.split_documents(docs_to_split)
    page_texts = {
        doc.metadata["page"]: doc.page_content
        for doc in page_docs
        if isinstance(doc.metadata.get("page"), int)
    }
    annotate_chunk_sections(chunks, page_texts)
    return chunks


from rag.section_filters import (
    DEFINITION_QUERY_RE,
    SECTION_IN_QUERY_RE,
    SECTION_SYMBOL_RE,
    _OBLIGATION_HINT_RE,
    _doc_key,
    _looks_like_section_one_definition,
    _section_matches,
    _text_looks_on_section,
    apply_hard_section_context_filter,
    apply_section_aware_retrieval,
    chunk_contains_section,
    chunk_content_matches_section,
    chunk_on_section,
    extract_section_text,
    extract_target_section,
    trim_document_to_section,
)
