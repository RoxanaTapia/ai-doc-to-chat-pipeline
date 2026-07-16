from pathlib import Path
import sys

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval_quality import (  # noqa: E402
    context_has_obligation_markers,
    context_sufficient_for_query,
    dedupe_similar_chunks,
    query_expects_obligations,
    sort_source_docs,
)


def test_query_expects_obligations_detects_prohibition_questions() -> None:
    assert query_expects_obligations(
        "What must the signatory agree not to do with Confidential Information?"
    )
    assert not query_expects_obligations("Does this agreement specify liquidated damages?")


def test_context_sufficient_when_obligation_markers_present() -> None:
    context = "I agree not to disclose, publish, or reveal Confidential Information."
    ok, gap = context_sufficient_for_query(
        "What must the signatory agree not to do?",
        context,
    )
    assert ok is True
    assert gap is None


def test_context_insufficient_when_obligation_question_lacks_markers() -> None:
    context = "PERSONAL NON-DISCLOSURE AGREEMENT\nName: John Doe"
    ok, gap = context_sufficient_for_query(
        "What must the signatory agree not to do?",
        context,
    )
    assert ok is False
    assert gap == "obligation_markers_missing"


def test_dedupe_similar_chunks_drops_near_duplicates() -> None:
    text = "1. Definition of Confidential Information\nCommercial value and labeled Confidential."
    candidates = [
        (Document(page_content=text, metadata={"page": 1}), 1.0),
        (Document(page_content=text + " extra", metadata={"page": 1}), 0.6),
        (Document(page_content="3. Obligations\nHold in strict confidence.", metadata={"page": 2}), 0.5),
    ]
    deduped = dedupe_similar_chunks(candidates, top_k=5)
    assert len(deduped) == 2
    pages = [doc.metadata["page"] for doc, _ in deduped]
    assert pages == [1, 2]


def test_dedupe_drops_tiny_boilerplate_when_longer_chunk_on_page() -> None:
    candidates = [
        (Document(page_content="Agreement.", metadata={"page": 2}), 1.0),
        (
            Document(
                page_content="3. Obligations of the Receiving Party\nHold in strict confidence.",
                metadata={"page": 2},
            ),
            0.8,
        ),
    ]
    deduped = dedupe_similar_chunks(candidates, top_k=5)
    assert len(deduped) == 1
    assert "Obligations" in deduped[0][0].page_content


def test_context_has_obligation_markers() -> None:
    assert context_has_obligation_markers("The party shall not disclose secrets.")
    assert not context_has_obligation_markers("7. Miscellaneous — laws of Spain.")


def test_sort_source_docs_on_section_before_off_section() -> None:
    off_high = Document(
        page_content="1. Definition of Confidential Information\nTrade secrets.",
        metadata={"page": 1, "similarity": 0.95},
    )
    on_low = Document(
        page_content="3. Obligations of the Receiving Party\nHold in confidence.",
        metadata={"page": 2, "similarity": 0.4},
    )
    on_high = Document(
        page_content="3. Obligations\nReturn or destroy materials.",
        metadata={"page": 2, "similarity": 0.8},
    )
    off_low = Document(
        page_content="4. Duration of Confidentiality\nFive years.",
        metadata={"page": 3, "similarity": 0.5},
    )
    ordered = sort_source_docs(
        [off_high, on_low, off_low, on_high],
        target_section="3",
    )
    assert ordered == [on_high, on_low, off_high, off_low]


def test_sort_source_docs_similarity_only_without_target() -> None:
    low = Document(page_content="a", metadata={"similarity": 0.2})
    high = Document(page_content="b", metadata={"similarity": 0.9})
    mid = Document(page_content="c", metadata={"similarity": 0.5})
    ordered = sort_source_docs([low, high, mid], target_section=None)
    assert ordered == [high, mid, low]


def test_sort_source_docs_missing_similarity_sorts_last() -> None:
    scored = Document(
        page_content="3. Obligations\nDuty.",
        metadata={"similarity": 0.3},
    )
    missing = Document(
        page_content="3. Obligations\nAnother duty.",
        metadata={},
    )
    ordered = sort_source_docs([missing, scored], target_section="3")
    assert ordered == [scored, missing]


def test_sort_source_docs_empty_input() -> None:
    assert sort_source_docs([], target_section="1") == []
