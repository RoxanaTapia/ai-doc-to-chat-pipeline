"""Wall-clock look-up timeout and question-stage logs."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
for _folder in (DEMO_DIR, SRC_DIR):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from ask_runtime import (  # noqa: E402
    ASK_LOGGER,
    AskTimeoutError,
    call_once,
    call_with_stage_timeout,
    log_ask_stage,
    open_generation,
    remaining_timeout,
    reset_call_state,
    run_question_stages,
    run_with_timeout,
)


def test_run_with_timeout_returns_quickly() -> None:
    assert run_with_timeout(lambda: "ok", 1.0) == "ok"


def test_run_with_timeout_raises_friendly_error() -> None:
    def hang() -> str:
        time.sleep(2)
        return "late"

    started = time.perf_counter()
    with pytest.raises(AskTimeoutError, match="Look-up exceeded"):
        run_with_timeout(hang, 0.05)
    assert time.perf_counter() - started < 1.0


def test_remaining_timeout_never_goes_non_positive() -> None:
    assert remaining_timeout(time.perf_counter() - 5.0, 1.0) == 0.1


def _capture_ask_logs() -> list[str]:
    lines: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    ASK_LOGGER.handlers.clear()
    ASK_LOGGER.addHandler(_Capture())
    ASK_LOGGER.setLevel(logging.INFO)
    ASK_LOGGER.propagate = False
    return lines


def test_log_ask_stage_writes_structured_fields() -> None:
    lines = _capture_ask_logs()
    log_ask_stage(
        "retrieval_done",
        question="How long does the confidentiality obligation last?",
        elapsed_ms=12.4,
        chunks=5,
    )
    joined = " ".join(lines)
    assert "stage=retrieval_done" in joined
    assert "elapsed_ms=12" in joined
    assert "question=" in joined


def test_run_question_stages_logs_each_step() -> None:
    lines = _capture_ask_logs()
    retrieved, answer, retrieval_ms, generation_ms = run_question_stages(
        question="How long does this agreement last?",
        retrieve_fn=lambda: ["chunk"],
        generate_fn=lambda: "3 years",
        timeout_seconds=1.0,
    )
    assert retrieved == ["chunk"]
    assert answer == "3 years"
    assert retrieval_ms >= 0.0
    assert generation_ms >= 0.0
    joined = " ".join(lines)
    assert "stage=start" in joined
    assert "stage=retrieval_done" in joined
    assert "stage=model_call_start" in joined
    assert "stage=model_call_end" in joined


def test_run_question_stages_logs_error_on_timeout() -> None:
    lines = _capture_ask_logs()

    def hang() -> str:
        time.sleep(2)
        return "late"

    started = time.perf_counter()
    with pytest.raises(AskTimeoutError):
        run_question_stages(
            question="How long does this agreement last?",
            retrieve_fn=hang,
            generate_fn=lambda: "unused",
            timeout_seconds=0.05,
        )
    assert time.perf_counter() - started < 1.0
    joined = " ".join(lines)
    assert "stage=start" in joined
    assert "stage=error" in joined
    assert "reason=AskTimeoutError" in joined


def test_anthropic_timeout_is_logged_without_an_end_line() -> None:
    """A hung model call logs the timeout stage and never the end stage."""
    lines = _capture_ask_logs()
    started = time.perf_counter()
    with pytest.raises(AskTimeoutError):
        call_with_stage_timeout(
            lambda: time.sleep(2),
            0.05,
            question="How long does this agreement last?",
            start_stage="anthropic_call_start",
            end_stage="anthropic_call_end",
            timeout_stage="anthropic_timeout",
        )
    assert time.perf_counter() - started < 1.0
    joined = " ".join(lines)
    assert "stage=anthropic_call_start" in joined
    assert "stage=anthropic_timeout" in joined
    assert "reason=AskTimeoutError" in joined
    assert "stage=anthropic_call_end" not in joined


def test_anthropic_call_logs_end_when_it_returns() -> None:
    lines = _capture_ask_logs()
    assert (
        call_with_stage_timeout(
            lambda: "3 years",
            1.0,
            question="How long does this agreement last?",
            start_stage="anthropic_call_start",
            end_stage="anthropic_call_end",
            timeout_stage="anthropic_timeout",
        )
        == "3 years"
    )
    joined = " ".join(lines)
    assert "stage=anthropic_call_start" in joined
    assert "stage=anthropic_call_end" in joined
    assert "stage=anthropic_timeout" not in joined


def test_call_once_runs_the_model_once_when_a_second_pass_joins() -> None:
    reset_call_state()
    key = ("session", "How long does this agreement last?")
    open_generation(key)
    started: list[int] = []
    release = threading.Event()

    def _slow() -> str:
        started.append(1)
        assert release.wait(2)
        return "3 years"

    first: dict[str, str] = {}
    second: dict[str, str] = {}
    opener = threading.Thread(
        target=lambda: first.setdefault("answer", call_once(key, _slow, 2))
    )
    joiner = threading.Thread(
        target=lambda: second.setdefault("answer", call_once(key, lambda: "second", 2))
    )
    opener.start()
    deadline = time.perf_counter() + 2
    while not started and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert started == [1]
    joiner.start()
    time.sleep(0.05)
    release.set()
    opener.join(timeout=2)
    joiner.join(timeout=2)
    assert first["answer"] == "3 years"
    assert second["answer"] == "3 years"
    assert started == [1]
    reset_call_state()
