"""Simple in-memory rate limiter for invite requests."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_events: int, window_seconds: int) -> None:
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = [t for t in self._events[key] if t >= cutoff]
            if len(bucket) >= self.max_events:
                self._events[key] = bucket
                return False
            bucket.append(now)
            self._events[key] = bucket
            return True
