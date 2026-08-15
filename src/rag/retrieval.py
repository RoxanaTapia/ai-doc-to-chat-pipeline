"""Hybrid search (FAISS + BM25 + RRF) and cross-encoder reranking."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document

from rag.chunking import apply_section_aware_retrieval, limit_chunks_per_page
from rag.citations import dedupe_similar_chunks

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    fetch_k: int
    early_page_max: int
    bm25_fetch_k: int
    rrf_k: int
    dense_weight: float
    bm25_weight: float
    reranker_top_n: int
    section_aware_enabled: bool
    section_aware_boost: float
    section_aware_min_chunks: int
    max_chunks_per_page: int
    dedupe_similar_chunks: bool
    dedupe_prefix_chars: int


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def doc_key(doc: Document) -> tuple:
    """Stable key for merging dense and sparse retrieval results."""
    stable_fingerprint = hashlib.blake2b(
        doc.page_content[:300].encode("utf-8", errors="ignore"),
        digest_size=12,
    ).hexdigest()
    return (
        doc.metadata.get("page", "N/A"),
        doc.metadata.get("start_index", "N/A"),
        stable_fingerprint,
    )


def distance_to_ui_similarity(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    minimum = min(scores)
    maximum = max(scores)
    if maximum - minimum <= 1e-12:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def build_bm25_index(chunks: list[Document]) -> tuple[object | None, str | None]:
    """Build a BM25 index for the current chunk set."""
    if not chunks:
        return None, "No chunks available for BM25 indexing."
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, (
            "Hybrid retrieval requires `rank-bm25`. "
            "Install it with `pip install -r requirements.txt`."
        )
    tokenized_corpus = [tokenize_for_bm25(doc.page_content) for doc in chunks]
    return BM25Okapi(tokenized_corpus), None


def semantic_search(
    query: str,
    *,
    vector_store,
    config: RetrievalConfig,
    limit: int | None = None,
) -> tuple[list[tuple[Document, float]], str, dict[str, int]]:
    """Dense retrieval with early-page preference."""
    effective_limit = max(1, int(limit or config.top_k))
    candidate_results = vector_store.similarity_search_with_score(query, k=config.fetch_k)

    early_page_results = []
    for doc, score in candidate_results:
        page = doc.metadata.get("page")
        if isinstance(page, int) and page <= config.early_page_max:
            early_page_results.append((doc, score))

    if early_page_results:
        selected = early_page_results[:effective_limit]
        mode = "semantic_early_page"
    else:
        selected = candidate_results[:effective_limit]
        mode = "semantic_global_fallback"

    return (
        [
            (doc, distance_to_ui_similarity(float(distance)))
            for doc, distance in selected
        ],
        mode,
        {
            "dense_candidates": len(candidate_results),
            "early_page_candidates": len(early_page_results),
            "selected_before_rerank": len(selected),
            "used_global_fallback": mode == "semantic_global_fallback",
        },
    )


def hybrid_rrf_search(
    query: str,
    *,
    vector_store,
    chunks: list[Document],
    bm25_index,
    config: RetrievalConfig,
    limit: int | None = None,
) -> tuple[list[tuple[Document, float]], str, str | None, dict[str, int]]:
    """Hybrid retrieval using dense + BM25 with Reciprocal Rank Fusion."""
    effective_limit = max(1, int(limit or config.top_k))
    dense_results = vector_store.similarity_search_with_score(query, k=config.fetch_k)

    if bm25_index is None:
        semantic_results, semantic_mode, semantic_diag = semantic_search(
            query, vector_store=vector_store, config=config, limit=effective_limit
        )
        return (
            semantic_results,
            f"{semantic_mode}_bm25_unavailable",
            None,
            {
                **semantic_diag,
                "sparse_candidates": 0,
                "fused_candidates": len(semantic_results),
            },
        )

    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        semantic_results, semantic_mode, semantic_diag = semantic_search(
            query, vector_store=vector_store, config=config, limit=effective_limit
        )
        return (
            semantic_results,
            f"{semantic_mode}_empty_query_tokens",
            None,
            {
                **semantic_diag,
                "sparse_candidates": 0,
                "fused_candidates": len(semantic_results),
            },
        )

    bm25_scores = bm25_index.get_scores(query_tokens)
    top_sparse_indices = [
        idx
        for idx, _score in sorted(
            enumerate(bm25_scores),
            key=lambda item: item[1],
            reverse=True,
        )[: config.bm25_fetch_k]
    ]

    fused_scores: dict[tuple, float] = defaultdict(float)
    doc_map: dict[tuple, Document] = {}

    for rank, (doc, _distance) in enumerate(dense_results, start=1):
        key = doc_key(doc)
        doc_map[key] = doc
        fused_scores[key] += config.dense_weight / (config.rrf_k + rank)

    for rank, chunk_idx in enumerate(top_sparse_indices, start=1):
        doc = chunks[chunk_idx]
        key = doc_key(doc)
        doc_map[key] = doc
        fused_scores[key] += config.bm25_weight / (config.rrf_k + rank)

    ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    top_candidates = [(doc_map[key], fused_scores[key]) for key in ranked_keys[:effective_limit]]
    normalized_scores = normalize_scores([score for _doc, score in top_candidates])
    normalized_candidates = [
        (doc, normalized_score)
        for (doc, _original_score), normalized_score in zip(top_candidates, normalized_scores)
    ]
    return normalized_candidates[:effective_limit], "hybrid_rrf", None, {
        "dense_candidates": len(dense_results),
        "sparse_candidates": len(top_sparse_indices),
        "fused_candidates": len(ranked_keys),
        "selected_before_rerank": len(top_candidates),
        "used_global_fallback": False,
    }


def first_stage_retrieval(
    query: str,
    *,
    strategy: str,
    vector_store,
    chunks: list[Document],
    bm25_index,
    bm25_warning: str | None,
    config: RetrievalConfig,
    candidate_limit: int,
) -> tuple[list[tuple[Document, float]], str, str | None, dict[str, int]]:
    if strategy == "hybrid":
        results, mode, warning, diag = hybrid_rrf_search(
            query,
            vector_store=vector_store,
            chunks=chunks,
            bm25_index=bm25_index,
            config=config,
            limit=candidate_limit,
        )
        return results, mode, warning or bm25_warning, diag

    semantic_results, mode, semantic_diag = semantic_search(
        query, vector_store=vector_store, config=config, limit=candidate_limit
    )
    return semantic_results, mode, None, semantic_diag


def apply_reranker(
    query: str,
    candidates: list[tuple[Document, float]],
    *,
    top_n: int,
    top_k: int,
    predict: Callable,
) -> tuple[list[tuple[Document, float]], str | None]:
    """Second-stage reranking. ``predict`` scores (query, passage) pairs."""
    if not candidates:
        return [], None
    try:
        effective_top_n = max(1, int(top_n))
        limited_candidates = candidates[:effective_top_n]
        pairs = [(query, doc.page_content[:2000]) for doc, _score in limited_candidates]
        raw_scores = predict(pairs)
        if hasattr(raw_scores, "tolist"):
            score_values = [float(score) for score in raw_scores.tolist()]
        else:
            score_values = [float(score) for score in raw_scores]
        normalized = normalize_scores(score_values)
        reranked = [
            (doc, normalized_score)
            for (doc, _original_score), normalized_score in zip(limited_candidates, normalized)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)

        final_results = reranked[:top_k]
        if len(final_results) < top_k:
            seen_keys = {doc_key(doc) for doc, _score in final_results}
            for doc, score in candidates:
                key = doc_key(doc)
                if key in seen_keys:
                    continue
                final_results.append((doc, score))
                seen_keys.add(key)
                if len(final_results) >= top_k:
                    break
        return final_results, None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return candidates[:top_k], (
            f"Reranker unavailable ({exc}). Falling back to first-stage ranking."
        )


def finalize_retrieval(
    query: str,
    *,
    candidate_pool: list[tuple[Document, float]],
    mode: str,
    enable_reranker: bool,
    rerank_predict: Callable | None,
    config: RetrievalConfig,
) -> tuple[list[tuple[Document, float]], str, str | None]:
    """Rerank, then apply section routing, page diversity, and dedupe."""
    pool = candidate_pool
    reranker_warning: str | None = None

    if enable_reranker and rerank_predict is not None:
        pool, reranker_warning = apply_reranker(
            query,
            candidate_pool,
            top_n=config.reranker_top_n,
            top_k=config.top_k,
            predict=rerank_predict,
        )
        mode = f"{mode}_reranked"

    section_warning: str | None = None
    if config.section_aware_enabled:
        pool, section_warning = apply_section_aware_retrieval(
            query,
            pool,
            top_k=config.top_k,
            boost=config.section_aware_boost,
            min_matching=config.section_aware_min_chunks,
        )
        if section_warning:
            mode = f"{mode}_section_aware"
    else:
        pool = pool[: config.top_k]

    if config.max_chunks_per_page > 0:
        pool = limit_chunks_per_page(
            pool, top_k=config.top_k, max_per_page=config.max_chunks_per_page
        )

    if config.dedupe_similar_chunks:
        pool = dedupe_similar_chunks(
            pool,
            top_k=config.top_k,
            prefix_chars=config.dedupe_prefix_chars,
        )
        mode = f"{mode}_deduped"

    combined_warning = " · ".join(
        part for part in (reranker_warning, section_warning) if part
    ) or None
    return pool[: config.top_k], mode, combined_warning
