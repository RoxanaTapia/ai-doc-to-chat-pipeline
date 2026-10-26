"""Three questions in one Streamlit session each paint a card."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
for _folder in (DEMO_DIR, SRC_DIR):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from ai_doc.rag.citations import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402
from page_copy import STARTER_CHIPS  # noqa: E402

TERM_QUESTION = STARTER_CHIPS[0]["question"]
LAW_QUESTION = STARTER_CHIPS[1]["question"]
FEE_QUESTION = STARTER_CHIPS[2]["question"]
ASK_KEY = "ask_submit"
TERM_CHIP_KEY = f"chip_{STARTER_CHIPS[0]['label']}"


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUB_EMBEDDINGS", "1")
    monkeypatch.setenv("DEMO_STUB_LLM", "1")
    monkeypatch.setenv("APP_PRESENTATION_MODE", "client")
    monkeypatch.setenv("USE_DUMMY_GENERATOR", "1")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _session_get(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


def _card_questions(at) -> list[str]:
    found: list[str] = []
    for block in at.markdown:
        value = block.value or ""
        if 'class="app-answer-q' not in value:
            continue
        text = value.split(">", 1)[-1].rsplit("<", 1)[0]
        found.append(text.replace("&#x27;", "'"))
    return found


def _ask(at, question: str) -> None:
    at.text_input[0].input(question)
    at.button(ASK_KEY).click()
    at.run(timeout=30)


def test_three_questions_in_one_session_each_render_a_card(stub_env) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert _session_get(at, "enable_reranker") is False
    assert not at.exception
    assert _session_get(at, "vector_store") is not None

    _ask(at, TERM_QUESTION)
    assert not at.exception
    _ask(at, FEE_QUESTION)
    assert not at.exception
    _ask(at, LAW_QUESTION)
    assert not at.exception

    messages = _session_get(at, "messages") or []
    assistants = [m for m in messages if m.get("role") == "assistant"]
    users = [m for m in messages if m.get("role") == "user"]
    assert len(users) == 3
    assert len(assistants) == 3
    assert all((msg.get("content") or "").strip() for msg in assistants)

    questions = _card_questions(at)
    assert len(questions) == 3
    assert TERM_QUESTION in questions
    assert FEE_QUESTION in questions
    assert LAW_QUESTION in questions

    contents = [msg.get("content") or "" for msg in assistants]
    assert any("3 years" in text for text in contents)
    assert any("Spain" in text for text in contents)
    assert any(
        INSUFFICIENT_CONTEXT_ANSWER in text or bool(msg.get("refusal"))
        for msg, text in zip(assistants, contents, strict=True)
    )
    assert any(msg.get("refusal") for msg in assistants)
    metrics = _session_get(at, "last_retrieval_metrics") or {}
    assert metrics.get("reranker_enabled") is False
    assert float(metrics.get("rerank_ms") or 0) == 0.0


def test_header_stays_intact_through_load_ask_and_answer(stub_env) -> None:
    """Privacy badge, status, and own-PDF stay on the page from load through answer.

    The data note lives in the lock details; CSS overlays it so this row cannot
    wrap the status under the note or shove "Try your own PDF" aside.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.session_state["sample_preload_failed"] = True
    at.run(timeout=30)
    assert not at.exception
    _assert_header_intact(at)
    assert not _chip_button_keys(at)

    at.session_state["sample_preload_failed"] = False
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is not None
    _assert_header_intact(at)
    assert TERM_CHIP_KEY in _chip_button_keys(at)
    blob = _markdown_blob(at)
    assert "How your data is handled" in blob or "Runs on this server" in blob
    assert "app-lock__line" in blob

    at.button(TERM_CHIP_KEY).click()
    at.run(timeout=30)
    assert not at.exception
    _assert_header_intact(at)
    assert not _chip_button_keys(at)
    messages = _session_get(at, "messages") or []
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert assistants and "3 years" in (assistants[0].get("content") or "")
    assert TERM_QUESTION in _card_questions(at)
    assert "Missing Submit Button" not in " ".join(
        str(getattr(err, "value", err)) for err in at.error
    )


def test_client_first_load_has_no_missing_submit_button(stub_env) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    error_text = " ".join(str(getattr(err, "value", err)) for err in at.error)
    assert "Missing Submit Button" not in error_text
    assert "missing submit" not in error_text.lower()
    assert _session_get(at, "vector_store") is not None
    assert at.button(ASK_KEY) is not None


def _chip_button_keys(at) -> set[str]:
    keys: set[str] = set()
    for btn in at.button:
        key = getattr(btn, "key", None)
        if key and str(key).startswith("chip_"):
            keys.add(str(key))
    return keys


def _markdown_blob(at) -> str:
    return "\n".join(block.value or "" for block in at.markdown)


def _header_parts_present(at) -> dict[str, bool]:
    """The header row must keep its three cells through load, ask, and answer."""
    blob = _markdown_blob(at)
    own_keys = {str(getattr(btn, "key", "") or "") for btn in at.button}
    return {
        "toolbar": "app-toolbar-mark" in blob,
        "lock": 'class="app-lock"' in blob or "app-lock" in blob,
        "lock_line": "app-lock__line" in blob,
        "status": "app-status" in blob,
        "own_pdf": "own_pdf_toggle" in own_keys,
        "ask": ASK_KEY in own_keys,
    }


def _assert_header_intact(at) -> None:
    parts = _header_parts_present(at)
    missing = [name for name, present in parts.items() if not present]
    assert not missing, f"header missing {missing} in {parts}"


def _plain_markdown_values(at) -> list[str]:
    values: list[str] = []
    for block in at.markdown:
        value = (block.value or "").strip()
        if value and "<" not in value:
            values.append(value)
    return values


def _chip_label_visible(at) -> bool:
    """True when the starter-chip label or a chip button is on the page."""
    if _chip_button_keys(at):
        return True
    for block in at.markdown:
        value = block.value or ""
        # Theme CSS also mentions .app-chip-mark; ignore <style> blocks.
        if "<style" in value:
            continue
        if 'class="app-chips-label app-chip-mark"' in value:
            return True
        if "Try a starter question" in value:
            return True
    return False


def test_starter_chips_hidden_until_chat_ready(stub_env) -> None:
    """No chips before the document is indexed; they appear once chat is ready."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.session_state["sample_preload_failed"] = True
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is None
    assert not _chip_button_keys(at)
    assert not _chip_label_visible(at)
    assert at.button(ASK_KEY).disabled

    at.session_state["sample_preload_failed"] = False
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is not None
    assert TERM_CHIP_KEY in _chip_button_keys(at)
    assert len(_chip_button_keys(at)) == 3
    assert _chip_label_visible(at)
    assert not at.button(ASK_KEY).disabled

    at.button(TERM_CHIP_KEY).click()
    at.run(timeout=30)
    assert not at.exception
    messages = _session_get(at, "messages") or []
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert users and users[0]["content"] == TERM_QUESTION
    assert assistants and "3 years" in (assistants[0].get("content") or "")
    assert not _chip_button_keys(at)
    assert not _chip_label_visible(at)


def test_starter_chip_submits_and_answers(stub_env) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is not None
    assert TERM_CHIP_KEY in _chip_button_keys(at)

    at.button(TERM_CHIP_KEY).click()
    at.run(timeout=30)
    assert not at.exception

    messages = _session_get(at, "messages") or []
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert users and users[0]["content"] == TERM_QUESTION
    assert assistants and "3 years" in (assistants[0].get("content") or "")
    ask_box = _session_get(at, "ask_box", "")
    assert ask_box in {"", None}
    assert not _chip_button_keys(at)
    leaked = "\n".join(_plain_markdown_values(at))
    assert LAW_QUESTION not in leaked
    assert FEE_QUESTION not in leaked
    assert TERM_QUESTION in _card_questions(at)


def test_type_then_click_ask_once_answers(stub_env) -> None:
    """Type a question and click Ask once. The answer must land on that click.

    In the browser the first Ask click often only blurs the field (Streamlit
    commits ``st.text_input`` on blur/Enter, and drops the button press).
    AppTest ``input()`` is that commit-only rerun: no button click. A second
    ``click()`` must not be required.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is not None
    assert at.button(ASK_KEY) is not None
    assert not at.button(ASK_KEY).disabled

    at.text_input[0].input(LAW_QUESTION)
    at.run(timeout=30)
    assert not at.exception

    messages = _session_get(at, "messages") or []
    users = [m for m in messages if m.get("role") == "user"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert len(users) == 1
    assert users[0]["content"] == LAW_QUESTION
    assert len(assistants) == 1
    assert "Spain" in (assistants[0].get("content") or "")
    assert LAW_QUESTION in _card_questions(at)
    assert _session_get(at, "queued_question") in {None, ""}
    assert _session_get(at, "ask_box", "") in {"", None}


def test_client_first_ask_skips_cross_encoder(stub_env) -> None:
    """Live-pilot default: first question must not load BAAI/bge-reranker-base."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "enable_reranker") is False
    assert _session_get(at, "vector_store") is not None

    started = time.perf_counter()
    _ask(at, TERM_QUESTION)
    elapsed = time.perf_counter() - started
    assert not at.exception
    assert elapsed < 5.0
    messages = _session_get(at, "messages") or []
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert "3 years" in (assistants[0].get("content") or "")
    metrics = _session_get(at, "last_retrieval_metrics") or {}
    assert metrics.get("reranker_enabled") is False
    assert float(metrics.get("retrieval_ms") or 0) < 2000.0


def _page_badge_visible(at) -> bool:
    return any('class="app-page-badge"' in (block.value or "") for block in at.markdown)


def _show_page_buttons(at) -> list[str]:
    found: list[str] = []
    for btn in at.button:
        label = getattr(btn, "label", None) or ""
        key = str(getattr(btn, "key", "") or "")
        if label == "Show page" or key.startswith("preview_"):
            found.append(key or label)
    return found


def _inline_quote_visible(at) -> bool:
    return any('class="app-inline-quote"' in (block.value or "") for block in at.markdown)


def test_second_ask_clears_prior_sources_while_loading(
    stub_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second question must not keep the first answer's sources at full contrast.

    Delays generation so the loading window is observable. AppTest does not
    refresh its element tree mid-script, so the test watches session flags and
    source-render calls: history is cleared before the model runs, and no page
    badge / quote / Show page is painted again until the new answer lands.
    """
    from streamlit.testing.v1 import AppTest

    import ai_doc.rag as rag
    import ai_doc.rag.generation as generation
    import sources_panel

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    assert _session_get(at, "vector_store") is not None

    _ask(at, TERM_QUESTION)
    assert not at.exception
    assert _page_badge_visible(at)
    assert _inline_quote_visible(at)
    assert _show_page_buttons(at)
    assert TERM_QUESTION in _card_questions(at)
    assert _session_get(at, "_answer_history_cleared") is False

    source_paint_events: list[str] = []
    real_inline = sources_panel._render_inline_sources
    real_preview = sources_panel._render_page_preview
    real_stream = generation.generate_answer_stream
    loading_snapshots: list[dict[str, object]] = []

    def _track_inline(*args, **kwargs):
        source_paint_events.append("inline")
        return real_inline(*args, **kwargs)

    def _track_preview(*args, **kwargs):
        source_paint_events.append("show_page")
        return real_preview(*args, **kwargs)

    def _delayed_stream(*args, **kwargs):
        time.sleep(0.35)
        loading_snapshots.append(
            {
                "cleared": _session_get(at, "_answer_history_cleared"),
                "status": _session_get(at, "doc_status"),
                "status_detail": _session_get(at, "doc_status_detail"),
                "source_paints_before_answer": list(source_paint_events),
            }
        )
        yield from real_stream(*args, **kwargs)

    monkeypatch.setattr(sources_panel, "_render_inline_sources", _track_inline)
    monkeypatch.setattr(sources_panel, "_render_page_preview", _track_preview)
    monkeypatch.setattr(generation, "generate_answer_stream", _delayed_stream)
    monkeypatch.setattr(rag, "generate_answer_stream", _delayed_stream)

    paints_after_first = list(source_paint_events)
    _ask(at, LAW_QUESTION)
    assert not at.exception
    assert loading_snapshots, "expected a mid-generation snapshot during the second ask"
    mid = loading_snapshots[0]
    assert mid["cleared"] is True
    assert mid["status"] == "answering"
    assert mid["status_detail"] == "Looking through the document"
    assert mid["source_paints_before_answer"] == paints_after_first

    messages = _session_get(at, "messages") or []
    assistants = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistants) == 2
    questions = _card_questions(at)
    assert TERM_QUESTION in questions
    assert LAW_QUESTION in questions
    assert _page_badge_visible(at)
    assert _inline_quote_visible(at)
    assert _show_page_buttons(at)
    assert _session_get(at, "_answer_history_cleared") is False
    assert source_paint_events != paints_after_first


def _status_trace(at, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record doc_status transitions around generation and the post-ask rerun."""
    import streamlit as st

    import ai_doc.rag as rag
    import ai_doc.rag.generation as generation

    trace: list[str] = []
    real_stream = generation.generate_answer_stream
    real_rerun = st.rerun

    def _tracked_stream(*args, **kwargs):
        trace.append(f"gen_start:{_session_get(at, 'doc_status')}")
        yield from real_stream(*args, **kwargs)
        trace.append(f"gen_end:{_session_get(at, 'doc_status')}")

    def _tracked_rerun(*args, **kwargs):
        trace.append(f"rerun:{_session_get(at, 'doc_status')}")
        return real_rerun(*args, **kwargs)

    monkeypatch.setattr(generation, "generate_answer_stream", _tracked_stream)
    monkeypatch.setattr(rag, "generate_answer_stream", _tracked_stream)
    monkeypatch.setattr(st, "rerun", _tracked_rerun)
    return trace


def test_followup_answer_has_no_busy_gap(stub_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Busy status must stay answering through generation until cards land."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    _ask(at, TERM_QUESTION)
    assert not at.exception
    assert _session_get(at, "doc_status") == "ready"

    trace = _status_trace(at, monkeypatch)
    _ask(at, LAW_QUESTION)
    assert not at.exception

    assert any(item.startswith("gen_start:answering") for item in trace)
    assert any(item.startswith("gen_end:answering") for item in trace)
    # Post-ask success rerun must still be answering (not ready) — that was the gap.
    success_reruns = [item for item in trace if item.startswith("rerun:")]
    assert success_reruns, "expected a post-ask rerun"
    assert success_reruns[-1] == "rerun:answering"

    assert _session_get(at, "doc_status") == "ready"
    questions = _card_questions(at)
    assert TERM_QUESTION in questions
    assert LAW_QUESTION in questions
    assistants = [m for m in (_session_get(at, "messages") or []) if m.get("role") == "assistant"]
    assert len(assistants) == 2
    assert "Spain" in (assistants[-1].get("content") or "")


def test_followup_refusal_has_no_busy_gap(stub_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusal follow-ups must also keep answering until the refusal card lands."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    _ask(at, TERM_QUESTION)
    assert not at.exception

    trace = _status_trace(at, monkeypatch)
    _ask(at, FEE_QUESTION)
    assert not at.exception

    # Context-guard refusals may skip the model; the success rerun must still
    # leave status answering so the busy line survives until the card paints.
    success_reruns = [item for item in trace if item.startswith("rerun:")]
    assert success_reruns, f"expected a post-ask rerun; trace={trace}"
    assert success_reruns[-1] == "rerun:answering"
    assert any(
        item.startswith("gen_start:answering") or item == "rerun:answering" for item in trace
    )

    assert _session_get(at, "doc_status") == "ready"
    assistants = [m for m in (_session_get(at, "messages") or []) if m.get("role") == "assistant"]
    assert len(assistants) == 2
    assert assistants[-1].get("refusal") or INSUFFICIENT_CONTEXT_ANSWER in (
        assistants[-1].get("content") or ""
    )
    assert FEE_QUESTION in _card_questions(at)


def test_cards_never_paint_alongside_busy_line(
    stub_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prior cards stay cleared for the whole busy window (no overlap with busy)."""
    from streamlit.testing.v1 import AppTest

    import ai_doc.rag as rag
    import ai_doc.rag.generation as generation
    import sources_panel

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=30)
    at.run(timeout=30)
    assert not at.exception
    _ask(at, TERM_QUESTION)
    assert not at.exception
    assert _page_badge_visible(at)

    loading_snapshots: list[dict[str, object]] = []
    card_paints_while_cleared = 0
    real_stream = generation.generate_answer_stream
    real_card = sources_panel._render_answer_card
    real_inline = sources_panel._render_inline_sources
    source_paints = 0

    def _count_inline(*args, **kwargs):
        nonlocal source_paints
        source_paints += 1
        return real_inline(*args, **kwargs)

    def _guard_card(*args, **kwargs):
        nonlocal card_paints_while_cleared
        # History flag is True only between clear and the idle refill. Painting
        # a card in that window would put prior sources under the busy line.
        if _session_get(at, "_answer_history_cleared") is True:
            card_paints_while_cleared += 1
        return real_card(*args, **kwargs)

    def _delayed_stream(*args, **kwargs):
        time.sleep(0.2)
        loading_snapshots.append(
            {
                "cleared": _session_get(at, "_answer_history_cleared"),
                "status": _session_get(at, "doc_status"),
                "source_paints": source_paints,
                "card_paints_while_cleared": card_paints_while_cleared,
            }
        )
        yield from real_stream(*args, **kwargs)

    monkeypatch.setattr(sources_panel, "_render_inline_sources", _count_inline)
    monkeypatch.setattr(sources_panel, "_render_answer_card", _guard_card)
    monkeypatch.setattr(generation, "generate_answer_stream", _delayed_stream)
    monkeypatch.setattr(rag, "generate_answer_stream", _delayed_stream)

    paints_before = source_paints
    _ask(at, LAW_QUESTION)
    assert not at.exception
    assert loading_snapshots
    mid = loading_snapshots[0]
    assert mid["cleared"] is True
    assert mid["status"] == "answering"
    assert mid["source_paints"] == paints_before
    assert mid["card_paints_while_cleared"] == 0
    assert card_paints_while_cleared == 0
    assert _session_get(at, "doc_status") == "ready"
    assert _page_badge_visible(at)
