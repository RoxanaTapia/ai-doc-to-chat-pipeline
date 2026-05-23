"""Section detection for legal PDFs — R4-B metadata and R4-C retrieval routing."""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document

NUMBERED_CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+(.+)$")
SECTION_IN_QUERY_RE = re.compile(r"section\s+(\d+)", re.IGNORECASE)
SECTION_SYMBOL_RE = re.compile(r"§\s*(\d+)")
SECTION_LABEL_RE = re.compile(r"(?i)\bsection\s+(\d+)\b")


def extract_target_section(query: str) -> str | None:
    """Parse target section from queries like 'Section 3' or '§3'."""
    text = query or ""
    match = SECTION_IN_QUERY_RE.search(text)
    if match:
        return match.group(1)
    match = SECTION_SYMBOL_RE.search(text)
    if match:
        return match.group(1)
    return None


def find_section_headers(text: str) -> list[tuple[int, str, str]]:
    """Return sorted (char_offset, section_id, title_snippet) headers in page text."""
    headers: list[tuple[int, str, str]] = []
    seen_at: set[int] = set()
    for match in NUMBERED_CLAUSE_RE.finditer(text or ""):
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, match.group(1), match.group(2).strip()[:80]))
    for match in SECTION_LABEL_RE.finditer(text or ""):
        pos = match.start()
        if pos in seen_at:
            continue
        seen_at.add(pos)
        headers.append((pos, match.group(1), ""))
    headers.sort(key=lambda item: item[0])
    return headers


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
    """R4-B: attach metadata.section / section_title from page headers + chunk start."""
    header_cache: dict[int, list[tuple[int, str, str]]] = {}
    for chunk in chunks:
        page = chunk.metadata.get("page")
        page_text = page_texts.get(page, "") if isinstance(page, int) else ""
        if isinstance(page, int) and page not in header_cache:
            header_cache[page] = find_section_headers(page_text)

        headers = header_cache.get(page, []) if isinstance(page, int) else []
        start_index = int(chunk.metadata.get("start_index", 0) or 0)
        section, title = section_at_offset(headers, start_index)

        if not section:
            section, title = _section_from_chunk_start(chunk.page_content)

        if section:
            chunk.metadata["section"] = section
        if title:
            chunk.metadata["section_title"] = title
    return chunks


def _doc_key(doc: Document) -> tuple:
    fingerprint = hashlib.blake2b(
        doc.page_content[:300].encode("utf-8", errors="ignore"),
        digest_size=8,
    ).digest()
    return (doc.metadata.get("page"), fingerprint)


def apply_section_aware_retrieval(
    query: str,
    candidates: list[tuple[Document, float]],
    *,
    top_k: int,
    boost: float = 1.5,
    min_matching: int = 2,
) -> tuple[list[tuple[Document, float]], str | None]:
    """
    R4-C: when the query names a section, boost matching chunks and prefer them in top-k.

    Falls back to the boosted full ranking if too few tagged chunks exist.
    """
    target = extract_target_section(query)
    if not target or not candidates:
        return candidates[:top_k], None

    boosted: list[tuple[Document, float]] = []
    for doc, score in candidates:
        section = str(doc.metadata.get("section") or "")
        if section == target:
            boosted.append((doc, score * boost))
        else:
            boosted.append((doc, score))
    boosted.sort(key=lambda item: item[1], reverse=True)

    matching = [item for item in boosted if str(item[0].metadata.get("section") or "") == target]
    unknown = [item for item in boosted if not item[0].metadata.get("section")]
    off_section = [
        item
        for item in boosted
        if item[0].metadata.get("section")
        and str(item[0].metadata.get("section")) != target
    ]

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

    in_top = sum(
        1 for doc, _score in final if str(doc.metadata.get("section") or "") == target
    )
    warning = None
    if in_top < min_matching:
        warning = (
            f"Section {target}: only {in_top} tagged chunk(s) in top-{top_k}; "
            "showing best-effort matches."
        )
    return final[:top_k], warning


def chunk_on_section(doc: Document, target_section: str) -> bool:
    """Eval helper: use metadata.section when present, else text heuristics."""
    tagged = doc.metadata.get("section")
    if tagged is not None:
        return str(tagged) == str(target_section)
    return _text_looks_on_section(doc.page_content, target_section)


def _text_looks_on_section(chunk_text: str, target_section: str) -> bool:
    text = chunk_text or ""
    head = text[:400]
    if re.search(rf"(?:^|\n)\s*{re.escape(target_section)}\.\s", head):
        return True
    if re.search(rf"(?i)\bsection\s+{re.escape(target_section)}\b", head):
        return True
    if target_section == "1" and re.search(
        r"(?i)(?:confidential information.*(?:means|shall mean)|is defined as)",
        text,
    ):
        if not re.search(
            r"(?i)\b(?:section\s+[2-9]|[2-9])\.\s+(?:exclusion|obligation|term)",
            head,
        ):
            return True
    return False
