"""Wall-clock timeout and stage logs for one look-up."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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


# One in-flight model call per (session, question). A Streamlit rerun can stop
# the script after the worker has the answer; the next pass reads it here
# instead of calling the model again.
_CALL_LOCK = threading.Lock()
_GENERATION: dict[tuple[str, str], int] = {}
_STASH: dict[tuple[str, str], tuple[int, str]] = {}
_PENDING: dict[tuple[str, str], tuple[int, Future[Any]]] = {}
_POOLS: dict[tuple[str, str], ThreadPoolExecutor] = {}


def reset_call_state() -> None:
    """Drop stashed answers and pending calls. Tests use this between cases."""
    with _CALL_LOCK:
        pools = list(_POOLS.values())
        _GENERATION.clear()
        _STASH.clear()
        _PENDING.clear()
        _POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=False)


def _release_pool(key: tuple[str, str]) -> None:
    pool = _POOLS.pop(key, None)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=False)


def open_generation(key: tuple[str, str]) -> int:
    """Start a new attempt. A late worker from the previous attempt cannot stash."""
    with _CALL_LOCK:
        generation = _GENERATION.get(key, 0) + 1
        _GENERATION[key] = generation
        _STASH.pop(key, None)
        _PENDING.pop(key, None)
        _release_pool(key)
        return generation


def current_generation(key: tuple[str, str]) -> int | None:
    with _CALL_LOCK:
        return _GENERATION.get(key)


def remember_answer(key: tuple[str, str], generation: int, text: str) -> None:
    """Keep a finished answer even if the script dies before session state is written."""
    with _CALL_LOCK:
        if _GENERATION.get(key) == generation:
            _STASH[key] = (generation, text)


def peek_answer(key: tuple[str, str]) -> str | None:
    with _CALL_LOCK:
        item = _STASH.get(key)
        if item is None or _GENERATION.get(key) != item[0]:
            return None
        return item[1]


def generation_pending(key: tuple[str, str]) -> bool:
    with _CALL_LOCK:
        current = _PENDING.get(key)
        return current is not None and current[0] == _GENERATION.get(key)


def invalidate_answer(key: tuple[str, str]) -> None:
    """The look-up failed. Ignore a worker that finishes after the error is stored."""
    with _CALL_LOCK:
        _GENERATION[key] = _GENERATION.get(key, 0) + 1
        _STASH.pop(key, None)
        _PENDING.pop(key, None)
        _release_pool(key)


def drop_answer(key: tuple[str, str]) -> None:
    """The reply is in the transcript. The stash is no longer the source of truth."""
    with _CALL_LOCK:
        _STASH.pop(key, None)
        _PENDING.pop(key, None)
        _release_pool(key)


def call_once(key: tuple[str, str], fn: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run ``fn`` once per generation. A later pass waits on that same call."""
    with _CALL_LOCK:
        generation = _GENERATION.get(key)
        if generation is None:
            generation = _GENERATION.get(key, 0) + 1
            _GENERATION[key] = generation
            _STASH.pop(key, None)
        current = _PENDING.get(key)
        if current is None or current[0] != generation:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ask-model")
            future: Future[Any] = pool.submit(fn)
            _PENDING[key] = (generation, future)
            _POOLS[key] = pool
            waiting = future
        else:
            waiting = current[1]
    try:
        return waiting.result(timeout=timeout_seconds)
    except FuturesTimeout as exc:
        raise AskTimeoutError(f"Look-up exceeded {timeout_seconds:.0f} seconds") from exc


def handoff_model_result(question: str, answer: str) -> str:
    """Main-thread handoff after the worker returns and before session state is written.

    A rerun pending from the Ask click raises at the next session write and
    drops the value. Tests interrupt here. The worker has already stashed
    ``answer``, so the following pass can store it without a second model call.
    """
    del question
    return answer


def call_with_stage_timeout(
    fn: Callable[[], Any],
    timeout_seconds: float,
    *,
    question: str,
    start_stage: str,
    end_stage: str,
    timeout_stage: str,
) -> Any:
    """Run ``fn`` with a timeout and one log line for start, end, or timeout.

    Used around the Anthropic call, on the worker thread. A hung HTTP call logs
    the timeout stage and never the end stage. A script stopped by a rerun does
    not produce the timeout stage; the script logs ``interrupted`` or the next
    pass logs ``abandoned``.
    """
    started = time.perf_counter()
    log_ask_stage(start_stage, question=question)
    try:
        result = run_with_timeout(fn, timeout_seconds)
    except AskTimeoutError:
        log_ask_stage(
            timeout_stage,
            question=question,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            reason="AskTimeoutError",
        )
        raise
    log_ask_stage(
        end_stage,
        question=question,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return result


def run_model_once(
    key: tuple[str, str],
    fn: Callable[[], str],
    timeout_seconds: float,
    *,
    question: str,
    resuming: bool,
    retrieval_ms: float | None = None,
) -> str:
    """Return the model text once. A stashed answer is not a second model call."""
    stashed = peek_answer(key)
    if stashed is not None:
        if resuming:
            log_ask_stage("resume", question=question, source="stash")
        return stashed
    if not generation_pending(key):
        log_ask_stage("model_call_start", question=question, elapsed_ms=retrieval_ms)
    started = time.perf_counter()
    text = call_once(key, fn, timeout_seconds)
    answer = text if isinstance(text, str) else ""
    # Log before any session-state write. That write is where a pending Ask
    # rerun stops the script, and it does so without an Exception traceback.
    log_ask_stage(
        "model_call_end",
        question=question,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return handoff_model_result(question, answer)
