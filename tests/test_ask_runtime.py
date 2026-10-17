"""Wall-clock look-up timeout and question-stage logs."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ask_runtime import (  # noqa: E402
    ASK_LOGGER,
    AskTimeoutError,
    log_ask_stage,
    remaining_timeout,
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
