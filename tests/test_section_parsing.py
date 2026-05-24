from pathlib import Path
import sys

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sectioning import (  # noqa: E402
    annotate_chunk_sections,
    apply_section_aware_retrieval,
    extract_target_section,
    find_section_headers,
    section_at_offset,
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
    from sectioning import apply_hard_section_context_filter

    ranked = [
        (Document(page_content="2. Exclusions", metadata={"section": "2"}), 1.0),
        (Document(page_content="3. Obligations", metadata={"section": "3"}), 0.8),
        (Document(page_content="WHEREAS preamble", metadata={}), 0.7),
    ]
    corpus = ranked[1:] + [
        (Document(page_content="more sec 3 duties", metadata={"section": "3"}), 0.0),
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


def test_hard_context_filter_skipped_when_no_section_in_query() -> None:
    from sectioning import apply_hard_section_context_filter

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
