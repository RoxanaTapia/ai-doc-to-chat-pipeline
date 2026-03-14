from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import rag  # noqa: E402


def test_generate_answer_dummy_mode_returns_echo() -> None:
    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=True,
    )
    assert "Dummy answer:" in answer
    assert "What is the notice period?" in answer
    assert "Section A says notice period is 30 days." in answer


def test_generate_answer_real_mode_calls_ollama_wrapper(monkeypatch) -> None:
    captured = {}

    def _fake_generate(context: str, query: str) -> str:
        captured["context"] = context
        captured["query"] = query
        return "final from ollama"

    monkeypatch.setattr(rag, "_generate_with_ollama", _fake_generate)

    answer = rag.generate_answer(
        context="Clause 5: Non-compete applies for 3 years.",
        query="Summarize restrictions.",
        dummy_mode=False,
    )

    assert answer == "final from ollama"
    assert captured["query"] == "Summarize restrictions."
    assert "Clause 5" in captured["context"]


def test_generate_with_ollama_respects_config_and_prompt(monkeypatch) -> None:
    created = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def invoke(self, prompt):
            created["prompt"] = prompt
            return SimpleNamespace(content=" Generated answer. ")

    monkeypatch.setattr(rag, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(
        rag,
        "load_generation_config",
        lambda: {
            "model": "phi3:mini",
            "temperature": 0.55,
            "top_p": 0.9,
            "max_new_tokens": 128,
            "do_sample": False,
            "fallback_to_dummy_on_error": False,
            "num_ctx": 2048,
            "timeout_seconds": 30,
        },
    )
    monkeypatch.setattr(
        rag,
        "load_rag_prompt",
        lambda: "CTX={context}\nQ={question}\nA=",
    )

    output = rag._generate_with_ollama("Context text", "Question text")

    assert output == "Generated answer."
    assert created["kwargs"]["model"] == "phi3:mini"
    assert created["kwargs"]["num_predict"] == 128
    assert created["kwargs"]["num_ctx"] == 2048
    assert created["kwargs"]["timeout"] == 30
    assert created["kwargs"]["top_p"] == 0.9
    # do_sample=False should force deterministic temperature.
    assert created["kwargs"]["temperature"] == 0.0
    assert "CTX=Context text" in created["prompt"]
    assert "Q=Question text" in created["prompt"]


def test_generate_answer_falls_back_to_dummy_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rag,
        "_generate_with_ollama",
        lambda context, query: (_ for _ in ()).throw(ConnectionError("connection refused")),
    )
    monkeypatch.setattr(
        rag,
        "load_generation_config",
        lambda: {"fallback_to_dummy_on_error": True},
    )

    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=False,
    )

    assert "Ollama unavailable, falling back to dummy mode." in answer
    assert "Dummy answer:" in answer


def test_generate_answer_raises_structured_error_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rag,
        "_generate_with_ollama",
        lambda context, query: (_ for _ in ()).throw(RuntimeError("model not found")),
    )
    monkeypatch.setattr(
        rag,
        "load_generation_config",
        lambda: {"fallback_to_dummy_on_error": False},
    )

    try:
        rag.generate_answer(
            context="Section A says notice period is 30 days.",
            query="What is the notice period?",
            dummy_mode=False,
        )
    except RuntimeError as exc:
        assert "Ollama generation unavailable:" in str(exc)
        assert "model not found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when fallback is disabled.")

