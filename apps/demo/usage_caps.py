"""Sample-only uploads and model-call caps for the public demo.

Counts live in a small SQLite file so a Streamlit rerun does not reset them.
A question the document cannot answer never reaches this store.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ALLOW_UPLOAD_ENV = "DEMO_ALLOW_UPLOAD"
UPLOAD_MAX_MB_ENV = "DEMO_UPLOAD_MAX_MB"
VISITOR_HOURLY_CAP_ENV = "DEMO_VISITOR_HOURLY_CAP"
DAILY_CAP_ENV = "DEMO_DAILY_CAP"
USAGE_STORE_ENV = "DEMO_USAGE_STORE"

DEFAULT_UPLOAD_MAX_MB = 10
DEFAULT_VISITOR_HOURLY_CAP = 20
DEFAULT_DAILY_CAP = 200
_HOUR_SECONDS = 3600
_DAY_SECONDS = 86400
_KEEP_SECONDS = 2 * _DAY_SECONDS


@dataclass(frozen=True)
class ModelCallPermit:
    """Whether this pass may call the model, and whether it consumed a slot."""

    call_model: bool
    counted: bool
    blocked_reason: str | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def visitor_uploads_enabled() -> bool:
    """True only when DEMO_ALLOW_UPLOAD is turned on. Empty stays off."""
    return _env_bool(ALLOW_UPLOAD_ENV, False)


def upload_controls_open(*, operator: bool) -> bool:
    """Visitors upload only when the flag is on. A signed-in operator always can."""
    return bool(operator) or visitor_uploads_enabled()


def upload_max_mb() -> int:
    return _env_int(UPLOAD_MAX_MB_ENV, DEFAULT_UPLOAD_MAX_MB)


def upload_max_bytes() -> int:
    return upload_max_mb() * 1024 * 1024


def upload_limit_label() -> str:
    """Short size for the page, such as ``10 MB``."""
    return f"{upload_max_mb()} MB"


def upload_exceeds_cap(size_bytes: int) -> bool:
    return int(size_bytes) > upload_max_bytes()


def visitor_hourly_cap() -> int:
    return _env_int(VISITOR_HOURLY_CAP_ENV, DEFAULT_VISITOR_HOURLY_CAP)


def daily_cap() -> int:
    return _env_int(DAILY_CAP_ENV, DEFAULT_DAILY_CAP)


def usage_store_path() -> Path:
    """Where the demo remembers model calls. Empty uses ``data/usage.sqlite``."""
    raw = (os.getenv(USAGE_STORE_ENV) or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "data" / "usage.sqlite"


def visitor_address(headers: object | None) -> str | None:
    """Visitor address from X-Forwarded-For.

    The reverse proxy appends the address it saw, so the last address is the
    visitor. Earlier addresses can be supplied by the browser.
    """
    if not headers:
        return None
    raw = None
    try:
        items = headers.items()
    except AttributeError:
        return None
    for key, value in items:
        if str(key).lower() == "x-forwarded-for":
            raw = value
            break
    if raw is None:
        return None
    parts = [part.strip().strip('"') for part in str(raw).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return parts[-1]


def visitor_key(headers: object | None, session_id: str | None) -> str:
    """IP from the proxy, or this browser session when that header is absent."""
    address = visitor_address(headers)
    if address:
        return f"ip:{address}"
    session = (session_id or "").strip()
    if session:
        return f"session:{session}"
    return "session:unknown"


def _utc_day_start(moment: float) -> float:
    current = datetime.fromtimestamp(moment, timezone.utc)
    midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


class UsageStore:
    """SQLite counts for one visitor's hour and the demo's UTC day."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else usage_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_calls (
                attempt_id TEXT PRIMARY KEY,
                visitor_key TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS model_calls_created ON model_calls (created_at)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def hourly_count(self, visitor: str, *, now: float | None = None) -> int:
        moment = time.time() if now is None else float(now)
        with self._lock:
            return self._hourly_count(visitor, moment)

    def daily_count(self, *, now: float | None = None) -> int:
        moment = time.time() if now is None else float(now)
        with self._lock:
            return self._daily_count(moment)

    def reserve(
        self,
        visitor: str,
        attempt_id: str,
        *,
        now: float | None = None,
    ) -> tuple[bool, str | None]:
        """Record one model call when both caps allow it.

        The same ``attempt_id`` is recorded once, so a rerun of an in-flight
        call does not take a second slot. Returns ``(allowed, reason)``.
        """
        moment = time.time() if now is None else float(now)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._prune(moment)
                if self._has_attempt(attempt_id):
                    self._conn.commit()
                    return True, None
                if self._hourly_count(visitor, moment) >= visitor_hourly_cap():
                    self._conn.commit()
                    return False, "hourly"
                if self._daily_count(moment) >= daily_cap():
                    self._conn.commit()
                    return False, "daily"
                self._conn.execute(
                    "INSERT INTO model_calls (attempt_id, visitor_key, created_at) "
                    "VALUES (?, ?, ?)",
                    (attempt_id, visitor, moment),
                )
                self._conn.commit()
                return True, None
            except Exception:
                self._conn.rollback()
                raise

    def _hourly_count(self, visitor: str, moment: float) -> int:
        return self._count(
            "SELECT COUNT(*) FROM model_calls WHERE visitor_key = ? AND created_at >= ?",
            (visitor, moment - _HOUR_SECONDS),
        )

    def _daily_count(self, moment: float) -> int:
        start = _utc_day_start(moment)
        return self._count(
            "SELECT COUNT(*) FROM model_calls WHERE created_at >= ? AND created_at < ?",
            (start, start + _DAY_SECONDS),
        )

    def _has_attempt(self, attempt_id: str) -> bool:
        return (
            self._count(
                "SELECT COUNT(*) FROM model_calls WHERE attempt_id = ?",
                (attempt_id,),
            )
            > 0
        )

    def _prune(self, moment: float) -> None:
        self._conn.execute(
            "DELETE FROM model_calls WHERE created_at < ?",
            (moment - _KEEP_SECONDS,),
        )

    def _count(self, sql: str, params: tuple) -> int:
        row = self._conn.execute(sql, params).fetchone()
        return int(row[0] if row else 0)


_STORE: UsageStore | None = None
_STORE_PATH: str | None = None


def usage_store() -> UsageStore:
    """Process-wide store. A new path (tests, a new env value) opens a new file."""
    global _STORE, _STORE_PATH
    path = str(usage_store_path())
    if _STORE is None or _STORE_PATH != path:
        if _STORE is not None:
            _STORE.close()
        _STORE = UsageStore(Path(path))
        _STORE_PATH = path
    return _STORE


def reset_usage_store() -> None:
    """Drop the open store. Tests call this after pointing at a new file."""
    global _STORE, _STORE_PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _STORE_PATH = None


def begin_model_call(
    *,
    visitor_key: str,
    attempt_id: str,
    guard_skipped: bool,
    already_started: bool,
    now: float | None = None,
    store: UsageStore | None = None,
) -> ModelCallPermit:
    """Decide a model call. A guard skip is not a call and does not count."""
    if guard_skipped:
        return ModelCallPermit(call_model=False, counted=False)
    if already_started:
        return ModelCallPermit(call_model=True, counted=False)
    target = store if store is not None else usage_store()
    allowed, reason = target.reserve(visitor_key, attempt_id, now=now)
    if not allowed:
        return ModelCallPermit(call_model=False, counted=False, blocked_reason=reason)
    return ModelCallPermit(call_model=True, counted=True)
