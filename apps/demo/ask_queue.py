"""Queue one question so Enter and Ask cannot start two look-ups.

Streamlit delivers a typed question twice. Enter or blur commits the text box
(``on_change``), and the Ask button is a second trigger a fraction of a second
later. Both call ``queue_question``. The second call is ignored while that
text is still pending, or within a short window of the look-up.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

REPEAT_ASK_WINDOW_S = 2.0
_RECENT_KEY = "_last_queued_question"


def _recent(state: MutableMapping[str, Any]) -> tuple[str, float] | None:
    raw = state.get(_RECENT_KEY)
    if not isinstance(raw, tuple) or len(raw) != 2:
        return None
    text, at = raw
    if not isinstance(text, str):
        return None
    try:
        return text, float(at)
    except (TypeError, ValueError):
        return None


def duplicate_question(state: MutableMapping[str, Any], question: str, *, now: float) -> bool:
    """True when this text is already pending or was queued moments ago."""
    if (state.get("queued_question") or "").strip() == question:
        return True
    if (state.get("_ask_inflight") or "").strip() == question:
        return True
    recent = _recent(state)
    if recent is None:
        return False
    text, at = recent
    return text == question and (now - at) < REPEAT_ASK_WINDOW_S


def queue_question(
    state: MutableMapping[str, Any],
    text: str,
    *,
    now: float | None = None,
) -> bool:
    """Queue stripped text once. False when it is empty or a duplicate."""
    question = (text or "").strip()
    if not question:
        return False
    moment = time.monotonic() if now is None else now
    if duplicate_question(state, question, now=moment):
        state["_clear_ask_box"] = True
        return False
    state["queued_question"] = question
    state["_ask_inflight"] = question
    state[_RECENT_KEY] = (question, moment)
    state["_clear_ask_box"] = True
    return True


def finish_question(
    state: MutableMapping[str, Any],
    question: str,
    *,
    now: float | None = None,
) -> None:
    """The reply is stored. Keep a short window so a late second trigger is ignored."""
    text = (question or "").strip()
    state["_ask_inflight"] = ""
    if not text:
        return
    moment = time.monotonic() if now is None else now
    state[_RECENT_KEY] = (text, moment)
