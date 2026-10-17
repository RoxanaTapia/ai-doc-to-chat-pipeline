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
ASK_KEY = "FormSubmitter:ask_form-Ask"


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
