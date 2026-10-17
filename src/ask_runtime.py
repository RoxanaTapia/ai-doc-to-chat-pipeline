"""One-question retrieve-and-generate, with a wall-clock timeout."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

from langchain_core.documents import Document

from rag.chunking import apply_hard_section_context_filter
from rag.citations import (
    INSUFFICIENT_CONTEXT_ANSWER,
    assemble_context,
    context_sufficient_for_query,
)
from rag.generation import generate_answer
from rag.retrieval import RetrievalConfig, finalize_retrieval, first_stage_retrieval


class AskTimeoutError(TimeoutError):
    """The retrieve-and-generate step did not finish in time."""


def run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` in a worker thread; raise AskTimeoutError when it overruns."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout as exc:
            future.cancel()
            raise AskTimeoutError(
                f"Look-up exceeded {timeout_seconds:.0f} seconds"
            ) from exc


def compute_answer(
    query: str,
    *,
    vector_store: Any,
    chunks: list[Document],
    retrieval_strategy: str,
    retrieval_config: RetrievalConfig,
    top_k: int,
    reranker_top_n: int,
    enable_reranker: bool,
    rerank_predict: Callable | None,
    dummy_mode: bool,
    use_page_separators: bool,
    section_hard_filter: bool,
    section_context_min_chunks: int,
    context_sufficiency_guard: bool,
    bm25_index: Any,
    bm25_warning: str | None,
    generate_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Retrieve context and generate an answer. No Streamlit calls."""
    writer = generate_fn or generate_answer
    retrieval_error: BaseException | None = None
    generation_error: BaseException | None = None
    raw_results: list[tuple[Document, float]] = []
    retrieved_docs: list[Document] = []
    context = ""
    retrieval_warning: str | None = None
    retrieval_diag: dict[str, int] = {}
    retrieval_elapsed_ms = 0.0
    rerank_elapsed_ms = 0.0
    generation_elapsed_ms = 0.0
    mode = "retrieval_failed"
    answer: str | None = None

    retrieval_start = time.perf_counter()
    try:
        candidate_limit = max(top_k, reranker_top_n) if enable_reranker else top_k
        candidate_pool, mode, first_stage_warning, retrieval_diag = first_stage_retrieval(
            query,
            strategy=retrieval_strategy,
            vector_store=vector_store,
            chunks=chunks or [],
            bm25_index=bm25_index,
            bm25_warning=bm25_warning,
            config=retrieval_config,
            candidate_limit=candidate_limit,
        )
        rerank_start = time.perf_counter()
        raw_results, mode, reranker_warning, rerank_elapsed_ms = _finalize(
            query,
            candidate_pool=candidate_pool,
            mode=mode,
            enable_reranker=enable_reranker,
            rerank_predict=rerank_predict,
            config=retrieval_config,
            rerank_start=rerank_start,
        )
        retrieval_warning = reranker_warning or first_stage_warning
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0

        for doc, score in raw_results:
            doc.metadata["similarity"] = score

        context_results, hard_filter_warning = apply_hard_section_context_filter(
            query,
            raw_results,
            all_chunks=chunks,
            top_k=top_k,
            enabled=section_hard_filter,
            min_chunks=section_context_min_chunks,
        )
        if hard_filter_warning:
            retrieval_warning = " · ".join(
                part for part in (retrieval_warning, hard_filter_warning) if part
            )
            mode = f"{mode}_section_filtered"

        retrieved_docs = [doc for doc, _score in context_results]
        context = assemble_context(
            context_results,
            use_page_separators=use_page_separators,
        )

        context_ok, _gap = context_sufficient_for_query(query, context)
        if context_sufficiency_guard and not context_ok:
            answer = INSUFFICIENT_CONTEXT_ANSWER
            mode = f"{mode}_context_guard"
        else:
            generation_start = time.perf_counter()
            try:
                answer = writer(context=context, query=query, dummy_mode=dummy_mode)
            except Exception as exc:
                generation_error = exc
                answer = None
            generation_elapsed_ms = (time.perf_counter() - generation_start) * 1000.0
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        retrieval_error = exc
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0
        mode = "retrieval_failed"
        raw_results = []
        retrieved_docs = []
        context = ""
        answer = None

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "raw_results": raw_results,
        "context": context,
        "mode": mode,
        "warning": retrieval_warning,
        "diag": retrieval_diag,
        "retrieval_ms": retrieval_elapsed_ms,
        "rerank_ms": rerank_elapsed_ms,
        "generation_ms": generation_elapsed_ms,
        "retrieval_error": retrieval_error,
        "generation_error": generation_error,
    }


def _finalize(
    query: str,
    *,
    candidate_pool: list[tuple[Document, float]],
    mode: str,
    enable_reranker: bool,
    rerank_predict: Callable | None,
    config: RetrievalConfig,
    rerank_start: float,
) -> tuple[list[tuple[Document, float]], str, str | None, float]:
    rerank_elapsed_ms = 0.0
    pool, mode, warning = finalize_retrieval(
        query,
        candidate_pool=candidate_pool,
        mode=mode,
        enable_reranker=enable_reranker,
        rerank_predict=rerank_predict,
        config=config,
    )
    if enable_reranker and rerank_predict is not None:
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000.0
    return pool, mode, warning, rerank_elapsed_ms
