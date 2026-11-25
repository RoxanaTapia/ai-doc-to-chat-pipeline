#!/usr/bin/env python3
"""Time the first sample-NDA answer, cold and warm.

Cold: libraries are imported, then the starter question builds the embedding
model and the index as part of the ask.

Warm: ``warm_sample_index()`` runs first, which is what the container does
before it listens. The clock then covers only the question.

The generator is the dummy placeholder, so the times are the index and
retrieval wait, not a hosted model. Reranking stays off, matching the client
demo. Prints ``cold_s`` and ``warm_s``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How long does this agreement last?"


def _pythonpath() -> str:
    src = ROOT / "src"
    demo = ROOT / "apps" / "demo"
    current = os.environ.get("PYTHONPATH", "")
    parts = [str(src), str(demo)]
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def _retrieval_config():
    import yaml

    from ai_doc.rag.retrieval import RetrievalConfig

    with (ROOT / "configs" / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    rag = config.get("rag") or {}
    retrieval = rag.get("retrieval") or {}
    section = retrieval.get("section_aware") or {}
    top_k = int(rag.get("top_k", 4))
    fetch_k = int(rag.get("fetch_k", 50))
    return RetrievalConfig(
        top_k=top_k,
        fetch_k=fetch_k,
        early_page_max=int(rag.get("early_page_max", 20)),
        bm25_fetch_k=int(retrieval.get("bm25_fetch_k", fetch_k)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
        dense_weight=float(retrieval.get("dense_weight", 1.0)),
        bm25_weight=float(retrieval.get("bm25_weight", 1.0)),
        reranker_top_n=int((retrieval.get("reranker") or {}).get("top_n", 50)),
        section_aware_enabled=bool(section.get("enabled", True)),
        section_aware_boost=float(section.get("boost", 1.5)),
        section_aware_min_chunks=int(section.get("min_chunks", 2)),
        max_chunks_per_page=int(retrieval.get("max_chunks_per_page", 2)),
        dedupe_similar_chunks=bool(retrieval.get("dedupe_similar_chunks", True)),
        dedupe_prefix_chars=int(retrieval.get("dedupe_prefix_chars", 180)),
    ), bool(section.get("hard_context_filter", True)), int(section.get("context_min_chunks", 2))


def answer_question() -> str:
    """One client-style ask against the sample NDA. Builds the index if needed."""
    from ai_doc.rag.chunking import apply_hard_section_context_filter
    from ai_doc.rag.citations import (
        INSUFFICIENT_CONTEXT_ANSWER,
        assemble_context,
        context_sufficient_for_query,
    )
    from ai_doc.rag.generation import generate_answer
    from ai_doc.rag.retrieval import build_bm25_index, finalize_retrieval, first_stage_retrieval
    from sample_warm import index_settings, sample_index_for

    model_name, chunk_size, chunk_overlap, header_split = index_settings()
    bundle = sample_index_for(model_name, chunk_size, chunk_overlap, header_split)
    config, hard_filter, context_min = _retrieval_config()
    bm25_index, bm25_warning = build_bm25_index(bundle["chunks"])
    pool, mode, _warning, _diag = first_stage_retrieval(
        QUESTION,
        strategy="hybrid",
        vector_store=bundle["vector_store"],
        chunks=bundle["chunks"],
        bm25_index=bm25_index,
        bm25_warning=bm25_warning,
        config=config,
        candidate_limit=config.top_k,
    )
    ranked, _mode, _warning = finalize_retrieval(
        QUESTION,
        candidate_pool=pool,
        mode=mode,
        enable_reranker=False,
        rerank_predict=None,
        config=config,
    )
    context_rows, _note = apply_hard_section_context_filter(
        QUESTION,
        ranked,
        all_chunks=bundle["chunks"],
        top_k=config.top_k,
        enabled=hard_filter,
        min_chunks=context_min,
    )
    context = assemble_context(context_rows, use_page_separators=True)
    enough, _gap = context_sufficient_for_query(QUESTION, context)
    if not enough:
        return INSUFFICIENT_CONTEXT_ANSWER
    return generate_answer(context, QUESTION, dummy_mode=True)


def _child(mode: str) -> int:
    os.environ["USE_DUMMY_GENERATOR"] = "1"
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("DEMO_STUB_LLM", None)
    os.environ.pop("USE_STUB_EMBEDDINGS", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Import before the clock so cold time is the index load, not interpreter start.
    import sample_warm

    if mode == "warm":
        bundle = sample_warm.warm_sample_index()
        if bundle is None:
            print("warm start failed", file=sys.stderr)
            return 1
    started = time.perf_counter()
    text = answer_question()
    elapsed = time.perf_counter() - started
    if not (text or "").strip():
        print("empty answer", file=sys.stderr)
        return 1
    print(f"{elapsed:.3f}")
    print(text.splitlines()[0], file=sys.stderr)
    return 0


def _run(mode: str) -> tuple[float, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["MEASURE_CHILD"] = mode
    env["USE_DUMMY_GENERATOR"] = "1"
    env.pop("LLM_PROVIDER", None)
    env.pop("DEMO_STUB_LLM", None)
    env.pop("USE_STUB_EMBEDDINGS", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", mode],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    seconds = float(completed.stdout.strip().splitlines()[-1])
    preview = (completed.stderr or "").strip().splitlines()
    return seconds, preview[-1] if preview else ""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--child"]:
        return _child(args[1])
    cold_s, cold_preview = _run("cold")
    warm_s, warm_preview = _run("warm")
    print(f"cold_s={cold_s:.3f}")
    print(f"warm_s={warm_s:.3f}")
    print(f"cold_answer={cold_preview}")
    print(f"warm_answer={warm_preview}")
    print(f"question={QUESTION}")
    print("generator=dummy")
    print("reranker=off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
