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
    """Queue stripped text once. False when it is empty or a duplicate.

    Does not mark the look-up in flight. That happens when a script pass
    actually starts it, so the pass that queued the text is not treated as a
    resume of a run that never began.
    """
    question = (text or "").strip()
    if not question:
        return False
    moment = time.monotonic() if now is None else now
    if duplicate_question(state, question, now=moment):
        state["_clear_ask_box"] = True
        return False
    state["queued_question"] = question
    state[_RECENT_KEY] = (question, moment)
    state["_clear_ask_box"] = True
    return True


def note_started(state: MutableMapping[str, Any], question: str) -> None:
    """This pass is running the look-up. A later pass can see that it was abandoned."""
    state["_ask_inflight"] = (question or "").strip()


def _latest_user_index(messages: Any, question: str) -> int | None:
    if not isinstance(messages, list):
        return None
    found: int | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if (message.get("content") or "") == question:
            found = index
    return found


def reply_is_stored(messages: Any, question: str) -> bool:
    """True when the latest user turn of this exact text already has a reply.

    An older completed ask of the same text does not count. A new ask is the
    user message that still has no assistant after it.
    """
    text = (question or "").strip()
    start = _latest_user_index(messages, text)
    if start is None or not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in messages[start + 1 :]
    )


def user_turn_open(messages: Any, question: str) -> bool:
    """True when this text is the latest user turn and the reply is not stored yet."""
    text = (question or "").strip()
    if _latest_user_index(messages, text) is None:
        return False
    return not reply_is_stored(messages, text)


def finish_question(
    state: MutableMapping[str, Any],
    question: str,
    *,
    now: float | None = None,
) -> None:
    """The reply is stored. Drop the queue and keep a short duplicate window."""
    text = (question or "").strip()
    state["_ask_inflight"] = ""
    if (state.get("queued_question") or "").strip() == text:
        state.pop("queued_question", None)
    if not text:
        return
    moment = time.monotonic() if now is None else now
    state[_RECENT_KEY] = (text, moment)
