"""Wall-clock look-up timeout."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ask_runtime import AskTimeoutError, run_with_timeout  # noqa: E402


def test_run_with_timeout_returns_quickly() -> None:
    assert run_with_timeout(lambda: "ok", 1.0) == "ok"


def test_run_with_timeout_raises_friendly_error() -> None:
    def hang() -> str:
        time.sleep(2)
        return "late"

    with pytest.raises(AskTimeoutError, match="Look-up exceeded"):
        run_with_timeout(hang, 0.05)
