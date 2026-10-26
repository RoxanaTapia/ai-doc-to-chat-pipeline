"""Wall-clock timeout and stage logs for one look-up."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any

ASK_LOGGER = logging.getLogger("ask")


class AskTimeoutError(TimeoutError):
    """The retrieve-and-generate step did not finish in time."""


def ensure_ask_logging() -> None:
    """Write ask-stage lines to stderr so `docker compose logs app` shows them."""
    if ASK_LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    ASK_LOGGER.addHandler(handler)
    ASK_LOGGER.setLevel(logging.INFO)


def log_ask_stage(
    stage: str,
    *,
    question: str,
    elapsed_ms: float | None = None,
    **fields: Any,
) -> None:
    """One structured line per question stage (start, retrieval, model, error)."""
    ensure_ask_logging()
    preview = " ".join((question or "").split())[:80]
    parts = [f"stage={stage}", f"question={preview!r}"]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms:.0f}")
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={value}")
    ASK_LOGGER.info("ask %s", " ".join(parts))


def remaining_timeout(started_at: float, budget_seconds: float) -> float:
    """Seconds left on a wall-clock budget, never below a short positive slice."""
    leftover = float(budget_seconds) - (time.perf_counter() - started_at)
    return max(0.1, leftover)


def run_with_timeout(fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` in a worker thread; raise AskTimeoutError when it overruns.

    Do not wait for the worker on timeout. A hung model call must not keep the
    Streamlit script (and the looking-through spinner) blocked.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeout as exc:
        future.cancel()
        raise AskTimeoutError(f"Look-up exceeded {timeout_seconds:.0f} seconds") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_question_stages(
    *,
    question: str,
    retrieve_fn: Callable[[], Any],
    generate_fn: Callable[[], Any] | None,
    timeout_seconds: float,
) -> tuple[Any, Any, float, float]:
    """Retrieve, then generate, under one budget. Log each stage with elapsed ms."""
    started = time.perf_counter()
    log_ask_stage("start", question=question)
    try:
        retrieved = run_with_timeout(
            retrieve_fn, remaining_timeout(started, timeout_seconds)
        )
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        log_ask_stage("retrieval_done", question=question, elapsed_ms=retrieval_ms)
        if generate_fn is None:
            return retrieved, None, retrieval_ms, 0.0
        log_ask_stage("model_call_start", question=question, elapsed_ms=retrieval_ms)
        gen_started = time.perf_counter()
        answer = run_with_timeout(
            generate_fn, remaining_timeout(started, timeout_seconds)
        )
        generation_ms = (time.perf_counter() - gen_started) * 1000.0
        log_ask_stage("model_call_end", question=question, elapsed_ms=generation_ms)
        return retrieved, answer, retrieval_ms, generation_ms
    except Exception as exc:
        log_ask_stage(
            "error",
            question=question,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            reason=type(exc).__name__,
        )
        raise
