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
