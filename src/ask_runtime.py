"""Wall-clock timeout for one look-up."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any


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
