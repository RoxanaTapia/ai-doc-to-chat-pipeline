"""Sample NDA penalty question must take the sentinel refusal path."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunking import chunk_pages  # noqa: E402
from rag.citations import (  # noqa: E402
    INSUFFICIENT_CONTEXT_ANSWER,
    assemble_context,
    build_sources_payload,
    is_insufficient_answer,
)
from rag.config import load_rag_prompt  # noqa: E402
from rag.ingestion import extract_pdf  # noqa: E402
from rag.retrieval import build_bm25_index, tokenize_for_bm25  # noqa: E402

PENALTY_QUESTION = "What is the penalty fee if Beta Labs breaks this agreement?"
NEAR_MISS = "no fee is specified; Section 6 allows injunctive relief"
SAMPLE_NDA = PROJECT_ROOT / "docs" / "product" / "sample-nda.pdf"


def _retrieve_sample_nda(query: str, *, top_k: int = 5) -> list[Document]:
    extracted = extract_pdf(SAMPLE_NDA, enable_ocr=False)
    chunks = chunk_pages(
        extracted.page_docs,
        chunk_size=1000,
        chunk_overlap=200,
        split_on_legal_headers=True,
    )
    bm25, warning = build_bm25_index(chunks)
    assert bm25 is not None, warning
    tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in ranked[:top_k] if score > 0]


def _stub_generator(_context: str, _query: str) -> str:
    return INSUFFICIENT_CONTEXT_ANSWER


def test_prompt_forbids_adjacent_penalty_guess() -> None:
    template = load_rag_prompt()
    assert INSUFFICIENT_CONTEXT_ANSWER in template
    assert "Do not give partial or adjacent answers" in template
    assert NEAR_MISS in template
    assert "Use the exact sentinel sentence instead" in template


def test_sample_nda_penalty_question_uses_sentinel_refusal() -> None:
    retrieved = _retrieve_sample_nda(PENALTY_QUESTION)
    assert retrieved, "retrieval should still return nearby passages from the NDA"
    context = assemble_context(
        [(doc, float(idx)) for idx, doc in enumerate(retrieved, start=1)],
        use_page_separators=True,
    )
    answer = _stub_generator(context, PENALTY_QUESTION)

    assert answer == INSUFFICIENT_CONTEXT_ANSWER
    assert is_insufficient_answer(answer)
    assert not is_insufficient_answer(NEAR_MISS)
    assert (
        build_sources_payload(
            retrieved,
            query=PENALTY_QUESTION,
            answer=answer,
            display_max=3,
            preview_chars=280,
        )
        == []
    )
