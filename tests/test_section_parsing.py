from pathlib import Path
import sys

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sectioning import (  # noqa: E402
    annotate_chunk_sections,
    apply_hard_section_context_filter,
    apply_section_aware_retrieval,
    chunk_contains_section,
    chunk_on_section,
    dominant_section_in_chunk,
    extract_section_text,
    extract_target_section,
    find_legal_headers,
    find_section_headers,
    limit_chunks_per_page,
    section_at_offset,
    split_documents_by_legal_headers,
    trim_document_to_section,
)


NDA_PAGE_1 = """\
WHEREAS, the parties wish to share information.

1. Definition of Confidential Information
Confidential Information means any information with commercial value.

2. Exclusions from Confidential Information
Confidential Information does not include public information.
"""

NDA_PAGE_2 = """\
3. Obligations of the Receiving Party
The Receiving Party shall hold all Confidential Information in strict confidence.

4. Duration of Confidentiality
Obligations continue for five years from disclosure.
"""

BOUNDARY_CHUNK = """\
(d) is approved in writing by the Disclosing Party.

3. Obligations of the Receiving Party
The Receiving Party shall hold all Confidential Information in strict confidence.
(a) first duty
(b) second duty
(c) third duty
(d) fourth duty

4. Duration of Confidentiality
Obligations continue for five years from disclosure.
"""

HERE_PAGE_1 = """\
Name: John Doe
Project: Pilot

Purpose.
"Purpose" as used in this Agreement shall mean any work carried out for HERE.

Confidentiality.
"Confidential Information" shall mean technical and non-technical information.
I agree not to disclose, publish, or reveal Confidential Information to any third party.
I agree not to make any public announcement of this NDA.
I agree to refrain from taking copies of Confidential Information.

No Further Rights.
Nothing herein shall grant any ownership of the Confidential Information.
"""


def test_extract_target_section_parses_common_patterns() -> None:
    assert extract_target_section("What does Section 3 require?") == "3"
    assert extract_target_section("defined in section 1") == "1"
    assert extract_target_section("See § 7 for venue") == "7"
    assert extract_target_section("liquidated damages?") is None


def test_find_section_headers_on_numbered_clauses() -> None:
    headers = find_section_headers(NDA_PAGE_1)
    assert [item[1] for item in headers] == ["1", "2"]
    assert "Definition" in headers[0][2]


def test_section_at_offset_follows_nearest_header() -> None:
    headers = find_section_headers(NDA_PAGE_1)
    preamble_section, _ = section_at_offset(headers, 0)
    assert preamble_section is None
    section_one, title = section_at_offset(headers, headers[0][0] + 5)
    assert section_one == "1"
    assert "Definition" in (title or "")


def test_annotate_chunk_sections_tags_sections() -> None:
    chunks = [
        Document(
            page_content="Confidential Information means any information with commercial value.",
            metadata={"page": 1, "start_index": 80},
        ),
        Document(
            page_content="3. Obligations of the Receiving Party\nHold in strict confidence.",
            metadata={"page": 2, "start_index": 0},
        ),
    ]
    annotate_chunk_sections(chunks, {1: NDA_PAGE_1, 2: NDA_PAGE_2})
    assert chunks[0].metadata.get("section") == "1"
    assert chunks[1].metadata.get("section") == "3"


def test_dominant_section_tags_boundary_chunk_as_section_three() -> None:
    section, title = dominant_section_in_chunk(BOUNDARY_CHUNK)
    assert section == "3"
    assert "Obligations" in (title or "")

    chunks = [
        Document(
            page_content=BOUNDARY_CHUNK,
            metadata={"page": 2, "start_index": 120},
        ),
    ]
    annotate_chunk_sections(chunks, {2: NDA_PAGE_2})
    assert chunks[0].metadata.get("section") == "3"
    assert chunks[0].metadata.get("section_span") == "multi"


def test_extract_section_text_trims_at_next_header() -> None:
    trimmed = extract_section_text(BOUNDARY_CHUNK, "3")
    assert trimmed is not None
    assert trimmed.startswith("3. Obligations")
    assert "fourth duty" in trimmed
    assert "4. Duration" not in trimmed
    assert chunk_contains_section(trimmed, "3")
    assert not chunk_contains_section(trimmed, "4")


def test_apply_section_aware_retrieval_prefers_matching_section() -> None:
    candidates = [
        (Document(page_content="2. Exclusions", metadata={"section": "2"}), 0.9),
        (Document(page_content="3. Obligations", metadata={"section": "3"}), 0.7),
        (Document(page_content="WHEREAS preamble", metadata={}), 0.85),
        (Document(page_content="4. Duration", metadata={"section": "4"}), 0.6),
        (Document(page_content="more sec 3", metadata={"section": "3"}), 0.55),
    ]
    results, warning = apply_section_aware_retrieval(
        "What does Section 3 impose?",
        candidates,
        top_k=3,
        boost=1.5,
        min_matching=2,
    )
    sections = [doc.metadata.get("section") for doc, _score in results]
    assert sections.count("3") >= 2
    assert warning is None


def test_apply_section_aware_retrieval_unchanged_without_section_in_query() -> None:
    candidates = [
        (Document(page_content="a", metadata={"section": "1"}), 0.5),
        (Document(page_content="b", metadata={"section": "2"}), 0.9),
    ]
    results, warning = apply_section_aware_retrieval(
        "liquidated damages?",
        candidates,
        top_k=2,
    )
    assert warning is None
    assert [doc.page_content for doc, _score in results] == ["a", "b"]


def test_hard_context_filter_keeps_only_target_section() -> None:
    ranked = [
        (Document(page_content="2. Exclusions", metadata={"section": "2"}), 1.0),
        (Document(page_content="3. Obligations", metadata={"section": "3"}), 0.8),
        (Document(page_content="WHEREAS preamble", metadata={}), 0.7),
    ]
    corpus = ranked[1:] + [
        (
            Document(
                page_content="3. Obligations\nAdditional duty text from corpus.",
                metadata={"section": "3"},
            ),
            0.0,
        ),
    ]
    corpus_docs = [doc for doc, _ in corpus]

    filtered, warning = apply_hard_section_context_filter(
        "What does Section 3 require?",
        ranked,
        all_chunks=corpus_docs,
        top_k=5,
        min_chunks=2,
    )
    sections = {doc.metadata.get("section") for doc, _ in filtered}
    assert sections == {"3"}
    assert len(filtered) >= 2
    assert warning is None


def test_hard_context_filter_rejects_mis_tagged_section_four_content() -> None:
    mis_tagged = Document(
        page_content=(
            "continuing obligations from prior text.\n"
            "4. Duration of Confidentiality\n"
            "Obligations continue for five years from disclosure."
        ),
        metadata={"section": "3"},
    )
    valid = Document(
        page_content="3. Obligations of the Receiving Party\nHold in strict confidence.",
        metadata={"section": "3"},
    )
    ranked = [(mis_tagged, 1.0), (valid, 0.9)]

    filtered, _warning = apply_hard_section_context_filter(
        "What does Section 3 require?",
        ranked,
        all_chunks=[valid],
        top_k=5,
        min_chunks=1,
    )
    assert len(filtered) == 1
    assert "4. Duration" not in filtered[0][0].page_content
    assert filtered[0][0].page_content.startswith("3.")


def test_hard_context_filter_trims_multi_section_chunks() -> None:
    ranked = [(Document(page_content=BOUNDARY_CHUNK, metadata={"section": "3"}), 1.0)]
    filtered, warning = apply_hard_section_context_filter(
        "What does Section 3 require?",
        ranked,
        all_chunks=[],
        top_k=5,
        min_chunks=1,
    )
    assert warning is None
    assert len(filtered) == 1
    content = filtered[0][0].page_content
    assert content.startswith("3.")
    assert "4. Duration" not in content


def test_hard_context_filter_skipped_when_no_section_in_query() -> None:
    ranked = [
        (Document(page_content="damages", metadata={"section": "6"}), 1.0),
    ]
    filtered, warning = apply_hard_section_context_filter(
        "liquidated damages?",
        ranked,
        all_chunks=[],
        top_k=5,
    )
    assert filtered == ranked
    assert warning is None


def test_chunk_on_section_uses_content_not_metadata_alone() -> None:
    mis_tagged = Document(
        page_content="4. Duration of Confidentiality\nFive years.",
        metadata={"section": "3"},
    )
    assert chunk_on_section(mis_tagged, "3") is False
    assert chunk_on_section(
        Document(page_content="3. Obligations\nDuty text.", metadata={}),
        "3",
    )


def test_limit_chunks_per_page_caps_same_page_results() -> None:
    candidates = [
        (Document(page_content="a", metadata={"page": 1}), 1.0),
        (Document(page_content="b", metadata={"page": 1}), 0.9),
        (Document(page_content="c", metadata={"page": 2}), 0.8),
        (Document(page_content="d", metadata={"page": 1}), 0.7),
        (Document(page_content="e", metadata={"page": 3}), 0.6),
    ]
    limited = limit_chunks_per_page(candidates, top_k=5, max_per_page=2)
    pages = [doc.metadata["page"] for doc, _ in limited]
    assert pages.count(1) == 2
    assert len(limited) == 4


def test_trim_document_to_section_returns_copy() -> None:
    doc = Document(page_content=BOUNDARY_CHUNK, metadata={"page": 2, "section": "3"})
    trimmed = trim_document_to_section(doc, "3")
    assert trimmed is not None
    assert trimmed.page_content != doc.page_content
    assert trimmed.metadata["section"] == "3"


def test_find_legal_headers_on_title_style_clauses() -> None:
    headers = find_legal_headers(HERE_PAGE_1)
    titles = [item[2] for item in headers]
    assert "Purpose" in titles
    assert "Confidentiality" in titles
    assert "No Further Rights" in titles


def test_split_documents_by_legal_headers_isolates_confidentiality() -> None:
    page_docs = [Document(page_content=HERE_PAGE_1, metadata={"page": 1})]
    sections = split_documents_by_legal_headers(page_docs)
    confidentiality = [
        doc for doc in sections if doc.metadata.get("section_title") == "Confidentiality"
    ]
    assert len(confidentiality) == 1
    text = confidentiality[0].page_content
    assert text.startswith("Confidentiality.")
    assert "agree not to disclose" in text
    assert "agree not to make any public announcement" in text
    assert "No Further Rights" not in text


def test_chunk_contains_section_matches_title_slug() -> None:
    doc = Document(
        page_content="Confidentiality.\nI agree not to disclose information.",
        metadata={"section": "confidentiality", "section_title": "Confidentiality"},
    )
    assert chunk_contains_section(doc.page_content, "confidentiality")
    assert doc.metadata.get("section") == "confidentiality"

