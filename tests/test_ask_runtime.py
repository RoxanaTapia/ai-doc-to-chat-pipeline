"""Timeout-bounded ask helper: consecutive looks-ups must not hang."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ask_runtime import AskTimeoutError, compute_answer, run_with_timeout  # noqa: E402
from rag.citations import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402
from rag.retrieval import RetrievalConfig  # noqa: E402


def _config() -> RetrievalConfig:
    return RetrievalConfig(
        top_k=3,
        fetch_k=8,
        early_page_max=20,
        bm25_fetch_k=8,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        reranker_top_n=8,
        section_aware_enabled=False,
        section_aware_boost=1.0,
        section_aware_min_chunks=1,
        max_chunks_per_page=2,
        dedupe_similar_chunks=False,
        dedupe_prefix_chars=180,
    )


def test_run_with_timeout_returns_quickly() -> None:
    assert run_with_timeout(lambda: "ok", 1.0) == "ok"


def test_run_with_timeout_raises_friendly_error() -> None:
    def hang() -> str:
        time.sleep(2)
        return "late"

    with pytest.raises(AskTimeoutError, match="Look-up exceeded"):
        run_with_timeout(hang, 0.05)


def test_compute_answer_skips_writer_when_amount_is_missing() -> None:
    chunks = [
        Document(
            page_content="Section 4 - Term. The agreement lasts three years.",
            metadata={"page": 2},
        )
    ]

    class _Store:
        def similarity_search_with_score(self, query: str, k: int):
            return [(chunks[0], 0.1)]

    def _must_not_run(**kwargs) -> str:
        raise AssertionError("writer should not run for a fee question with no money")

    result = compute_answer(
        "What is the penalty fee if Beta Labs breaks this agreement?",
        vector_store=_Store(),
        chunks=chunks,
        retrieval_strategy="semantic",
        retrieval_config=_config(),
        top_k=3,
        reranker_top_n=8,
        enable_reranker=False,
        rerank_predict=None,
        dummy_mode=True,
        use_page_separators=False,
        section_hard_filter=False,
        section_context_min_chunks=1,
        context_sufficiency_guard=True,
        bm25_index=None,
        bm25_warning=None,
        generate_fn=_must_not_run,
    )
    assert result["generation_error"] is None
    assert result["retrieval_error"] is None
    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert "context_guard" in result["mode"]
