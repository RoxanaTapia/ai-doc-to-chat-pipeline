"""Sample-only mode, upload size, and model-call caps for the public demo."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DIR = PROJECT_ROOT / "apps" / "demo"
for _folder in (DEMO_DIR, SRC_DIR):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from ai_doc.rag.citations import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402
from page_copy import _CLIENT_COPY  # noqa: E402
from usage_caps import (  # noqa: E402
    DEFAULT_DAILY_CAP,
    DEFAULT_UPLOAD_MAX_MB,
    DEFAULT_VISITOR_HOURLY_CAP,
    UsageStore,
    begin_model_call,
    daily_cap,
    reset_usage_store,
    upload_controls_open,
    upload_exceeds_cap,
    upload_limit_label,
    upload_max_bytes,
    upload_max_mb,
    usage_store,
    visitor_address,
    visitor_hourly_cap,
    visitor_key,
    visitor_uploads_enabled,
)

CAP_COPY = (
    "This demo has reached its limit for now. "
    "Please try again later, or book a short call and I'll walk you through it."
)
FEE_QUESTION = "What's the penalty fee?"
TERM_QUESTION = "How long does this last?"
LAW_QUESTION = "Which law applies?"
ASK_KEY = "ask_submit"


def test_cap_copy_is_plain() -> None:
    assert _CLIENT_COPY["usage_cap"] == CAP_COPY
    assert "\u2014" not in _CLIENT_COPY["usage_cap"]
    assert "\u2014" not in _CLIENT_COPY["upload_too_large"]
    assert "\u2014" not in _CLIENT_COPY["upload_limit_hint"]


def test_sample_only_unless_upload_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_ALLOW_UPLOAD", raising=False)
    assert visitor_uploads_enabled() is False
    assert upload_controls_open(operator=False) is False
    assert upload_controls_open(operator=True) is True
    monkeypatch.setenv("DEMO_ALLOW_UPLOAD", "")
    assert visitor_uploads_enabled() is False
    monkeypatch.setenv("DEMO_ALLOW_UPLOAD", "true")
    assert visitor_uploads_enabled() is True
    assert upload_controls_open(operator=False) is True


def test_upload_size_cap_defaults_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_UPLOAD_MAX_MB", raising=False)
    assert upload_max_mb() == DEFAULT_UPLOAD_MAX_MB
    assert upload_limit_label() == "10 MB"
    assert upload_exceeds_cap(upload_max_bytes()) is False
    assert upload_exceeds_cap(upload_max_bytes() + 1) is True
    monkeypatch.setenv("DEMO_UPLOAD_MAX_MB", "2")
    assert upload_limit_label() == "2 MB"
    assert upload_exceeds_cap(2 * 1024 * 1024) is False
    assert upload_exceeds_cap(2 * 1024 * 1024 + 1) is True
    monkeypatch.setenv("DEMO_UPLOAD_MAX_MB", "lots")
    assert upload_max_mb() == DEFAULT_UPLOAD_MAX_MB


def test_visitor_address_uses_proxy_hop_then_session() -> None:
    forwarded = {"X-Forwarded-For": "1.1.1.1, 203.0.113.9"}
    assert visitor_address(forwarded) == "203.0.113.9"
    assert visitor_key(forwarded, "session-a") == "ip:203.0.113.9"
    assert visitor_key({"x-forwarded-for": " 203.0.113.8 "}, "session-a") == "ip:203.0.113.8"
    assert visitor_key({"X-Forwarded-For": " , "}, "session-b") == "session:session-b"
    assert visitor_key({}, "session-b") == "session:session-b"
    assert visitor_key(None, "  ") == "session:unknown"


def test_guard_skip_and_resume_do_not_count(tmp_path: Path) -> None:
    store = UsageStore(tmp_path / "caps.sqlite")
    now = 1_700_000_000.0
    skipped = begin_model_call(
        visitor_key="ip:203.0.113.9",
        attempt_id="skip",
        guard_skipped=True,
        already_started=False,
        now=now,
        store=store,
    )
    resumed = begin_model_call(
        visitor_key="ip:203.0.113.9",
        attempt_id="again",
        guard_skipped=False,
        already_started=True,
        now=now,
        store=store,
    )
    assert skipped.call_model is False
    assert skipped.counted is False
    assert resumed.call_model is True
    assert resumed.counted is False
    assert store.daily_count(now=now) == 0
    assert store.hourly_count("ip:203.0.113.9", now=now) == 0
    store.close()


def test_visitor_hourly_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "2")
    monkeypatch.setenv("DEMO_DAILY_CAP", "100")
    store = UsageStore(tmp_path / "hour.sqlite")
    now = 10_000.0
    visitor = "ip:203.0.113.9"
    other = "ip:198.51.100.4"
    first = store.reserve(visitor, "a1", now=now)
    second = store.reserve(visitor, "a2", now=now)
    blocked = store.reserve(visitor, "a3", now=now)
    other_ok = store.reserve(other, "b1", now=now)
    repeat = store.reserve(visitor, "a1", now=now)
    assert first == (True, None)
    assert second == (True, None)
    assert blocked == (False, "hourly")
    assert other_ok == (True, None)
    assert repeat == (True, None)
    assert store.hourly_count(visitor, now=now) == 2
    # A call exactly one hour ago still counts. One second older does not.
    edge = UsageStore(tmp_path / "edge.sqlite")
    edge.reserve(visitor, "old", now=now - 3600)
    edge.reserve(visitor, "older", now=now - 3601)
    assert edge.hourly_count(visitor, now=now) == 1
    later = store.reserve(visitor, "a4", now=now + 3601)
    assert later == (True, None)
    store.close()
    edge.close()


def test_daily_cap_is_shared_and_resets_next_utc_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "100")
    monkeypatch.setenv("DEMO_DAILY_CAP", "2")
    store = UsageStore(tmp_path / "day.sqlite")
    now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc).timestamp()
    yesterday = datetime(2026, 11, 25, 23, 30, tzinfo=timezone.utc).timestamp()
    store.reserve("ip:1", "yesterday", now=yesterday)
    assert store.daily_count(now=now) == 0
    assert store.reserve("ip:1", "today-1", now=now) == (True, None)
    assert store.reserve("ip:2", "today-2", now=now) == (True, None)
    blocked = store.reserve("ip:3", "today-3", now=now)
    assert blocked == (False, "daily")
    assert store.daily_count(now=now) == 2
    tomorrow = datetime(2026, 11, 27, 0, 5, tzinfo=timezone.utc).timestamp()
    assert store.reserve("ip:3", "tomorrow", now=tomorrow) == (True, None)
    store.close()


def test_counts_survive_a_new_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_DAILY_CAP", "10")
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "10")
    path = tmp_path / "persist.sqlite"
    now = datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc).timestamp()
    first = UsageStore(path)
    assert first.reserve("session:abc", "one", now=now) == (True, None)
    first.close()
    second = UsageStore(path)
    assert second.daily_count(now=now) == 1
    assert second.hourly_count("session:abc", now=now) == 1
    second.close()


def test_empty_caps_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_VISITOR_HOURLY_CAP", raising=False)
    monkeypatch.delenv("DEMO_DAILY_CAP", raising=False)
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "")
    monkeypatch.setenv("DEMO_DAILY_CAP", " ")
    assert visitor_hourly_cap() == DEFAULT_VISITOR_HOURLY_CAP
    assert daily_cap() == DEFAULT_DAILY_CAP
    monkeypatch.delenv("DEMO_VISITOR_HOURLY_CAP", raising=False)
    monkeypatch.delenv("DEMO_DAILY_CAP", raising=False)
    assert visitor_hourly_cap() == 20
    assert daily_cap() == 200


def test_enforce_sample_only_drops_a_visitor_pdf() -> None:
    import session_upload
    from session_upload import enforce_sample_only

    class _State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def pop(self, key, default=None):
            return dict.pop(self, key, default)

    fake = _State(
        vector_store=object(),
        chunks=["visitor"],
        current_file="contract.pdf",
        last_processed_name="contract.pdf",
        last_processed_hash="abc",
        uploaded_pdf_bytes=b"%PDF",
        uploaded_pdf_name="contract.pdf",
        uploaded_pdf_hash="abc",
        bm25_state=None,
        indexed_doc_stats={"file_name": "contract.pdf"},
        last_generation_error=None,
        sample_preload_failed=False,
        uploader_key_version=1,
        own_pdf_mode=True,
        show_own_pdf=True,
        messages=[{"role": "user", "content": "hi"}],
        last_query="hi",
        last_retrieved_docs=[],
        last_retrieval_mode=None,
        last_raw_results=[],
        last_context="",
        last_answer=None,
        last_retrieval_metrics=None,
        last_response_timing=None,
        doc_status="ready",
        doc_status_detail="Contract",
        dev_index_logs=None,
    )
    previous = session_upload.st.session_state
    session_upload.st.session_state = fake
    try:
        enforce_sample_only(upload_open=True)
        assert fake.own_pdf_mode is True
        assert fake.vector_store is not None
        enforce_sample_only(upload_open=False)
        assert fake.own_pdf_mode is False
        assert fake.show_own_pdf is False
        assert fake.vector_store is None
        assert "uploaded_pdf_name" not in fake
        fake.uploaded_pdf_name = "sample-nda.pdf"
        fake.current_file = "sample-nda.pdf"
        fake.vector_store = object()
        fake.own_pdf_mode = True
        enforce_sample_only(upload_open=False)
        assert fake.own_pdf_mode is False
        assert fake.vector_store is not None
    finally:
        session_upload.st.session_state = previous


def test_limits_are_documented_without_internal_words() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    running = (PROJECT_ROOT / "docs" / "operators" / "running.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for name in (
        "DEMO_ALLOW_UPLOAD",
        "DEMO_UPLOAD_MAX_MB",
        "DEMO_VISITOR_HOURLY_CAP",
        "DEMO_DAILY_CAP",
        "DEMO_USAGE_STORE",
    ):
        assert f"{name}=" in example
        assert name in running
    assert "DEMO_ALLOW_UPLOAD=\n" in example
    assert "PDFs up to the size shown on the page" in readme or "size shown on the page" in readme
    assert "enough questions for the hour or the day" in readme
    blob = "\n".join((example, running, readme)).lower()
    assert "ttl" not in blob
    assert "simulation" not in blob
    assert "\u2014" not in running.split("## Demo limits", 1)[1].split("## ", 1)[0]


def _visible_text(at) -> str:
    parts = [(block.value or "") for block in at.markdown]
    captions = getattr(at, "caption", None)
    if captions is not None:
        parts.extend((block.value or "") for block in captions)
    return "\n".join(parts)


def _button_keys(at) -> set[str]:
    return {str(getattr(btn, "key", "") or "") for btn in at.button}


def _ask(at, question: str) -> None:
    at.text_input[0].input(question)
    at.button(ASK_KEY).click()
    at.run(timeout=60)


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUB_EMBEDDINGS", "1")
    monkeypatch.setenv("DEMO_STUB_LLM", "1")
    monkeypatch.setenv("APP_PRESENTATION_MODE", "client")
    monkeypatch.setenv("USE_DUMMY_GENERATOR", "1")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_visitor_sees_sample_only_until_upload_is_enabled(stub_env, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=60)
    at.run(timeout=60)
    assert not at.exception
    assert "own_pdf_toggle" not in _button_keys(at)
    assert "PDFs up to" not in _visible_text(at)

    monkeypatch.setenv("DEMO_ALLOW_UPLOAD", "true")
    opened = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=60)
    opened.run(timeout=60)
    assert not opened.exception
    assert "own_pdf_toggle" in _button_keys(opened)
    opened.button("own_pdf_toggle").click()
    opened.run(timeout=60)
    assert not opened.exception
    blob = _visible_text(opened)
    assert "PDFs up to 10 MB." in blob


def test_hourly_cap_blocks_the_next_model_call_and_ignores_guard(
    stub_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "1")
    monkeypatch.setenv("DEMO_DAILY_CAP", "100")
    reset_usage_store()
    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=60)
    at.run(timeout=60)
    assert not at.exception

    _ask(at, FEE_QUESTION)
    assert not at.exception
    assert usage_store().daily_count() == 0
    assert at.session_state["messages"][-1]["content"] == INSUFFICIENT_CONTEXT_ANSWER

    _ask(at, TERM_QUESTION)
    assert not at.exception
    assert usage_store().daily_count() == 1
    assert "3 years" in at.session_state["messages"][-1]["content"]

    _ask(at, LAW_QUESTION)
    assert not at.exception
    assert usage_store().daily_count() == 1
    answer = at.session_state["messages"][-1]["content"]
    assert answer == CAP_COPY
    assert "Spain" not in answer
    visible = _visible_text(at)
    assert "This demo has reached its limit for now." in visible
    assert "book a short call" in visible


def test_daily_cap_blocks_every_visitor(stub_env, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "100")
    monkeypatch.setenv("DEMO_DAILY_CAP", "1")
    reset_usage_store()
    at = AppTest.from_file(str(DEMO_DIR / "app.py"), default_timeout=60)
    at.run(timeout=60)
    assert not at.exception
    _ask(at, TERM_QUESTION)
    assert not at.exception
    assert usage_store().daily_count() == 1
    _ask(at, LAW_QUESTION)
    assert not at.exception
    assert usage_store().daily_count() == 1
    assert at.session_state["messages"][-1]["content"] == CAP_COPY
    visible = _visible_text(at)
    assert "This demo has reached its limit for now." in visible
    assert "book a short call" in visible
