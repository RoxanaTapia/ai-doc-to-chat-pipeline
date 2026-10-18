"""Three questions in one Streamlit session each paint a card."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from page_copy import STARTER_CHIPS  # noqa: E402
from rag.citations import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402

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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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
        if "app-chip-mark" in value or "Try a starter question" in value:
            return True
    return False


def test_starter_chips_hidden_until_chat_ready(stub_env) -> None:
    """No chips before the document is indexed; they appear once chat is ready."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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

    at = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30)
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
