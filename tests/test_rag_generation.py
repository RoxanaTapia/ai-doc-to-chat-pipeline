import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ai_doc.rag as rag  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_llm_provider_env(monkeypatch) -> None:
    """Keep legacy dummy_mode mapping tests independent of the host env."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("DEMO_STUB_LLM", raising=False)
    rag.load_generation_config.cache_clear()


def test_generate_answer_dummy_mode_returns_placeholder() -> None:
    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=True,
    )
    assert "no AI model is running here" in answer
    assert "live pilot" in answer


def test_generate_answer_real_mode_calls_ollama_wrapper(monkeypatch) -> None:
    captured = {}

    def _fake_generate(context: str, query: str, settings=None) -> str:
        captured["context"] = context
        captured["query"] = query
        captured["settings"] = settings
        return "final from ollama"

    monkeypatch.setattr(rag.providers, "_generate_with_ollama", _fake_generate)

    answer = rag.generate_answer(
        context="Clause 5: Non-compete applies for 3 years.",
        query="Summarize restrictions.",
        dummy_mode=False,
    )

    assert answer == "final from ollama"
    assert captured["query"] == "Summarize restrictions."
    assert "Clause 5" in captured["context"]
    assert isinstance(captured["settings"], dict)


def test_generate_answer_passes_settings_to_ollama_wrapper(monkeypatch) -> None:
    captured = {}

    def _fake_generate(context: str, query: str, settings=None) -> str:
        captured["settings"] = settings
        return "ok"

    monkeypatch.setattr(rag.providers, "_generate_with_ollama", _fake_generate)
    monkeypatch.setattr(
        rag.providers,
        "load_generation_config",
        lambda: {
            "model": "phi3:mini",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 64,
            "do_sample": False,
            "fallback_to_dummy_on_error": False,
            "num_ctx": 2048,
            "timeout_seconds": 30,
        },
    )

    answer = rag.generate_answer(
        context="Sample context",
        query="Sample query",
        dummy_mode=False,
    )

    assert answer == "ok"
    assert captured["settings"]["model"] == "phi3:mini"
    assert captured["settings"]["max_new_tokens"] == 64


def test_generate_with_ollama_respects_config_and_prompt(monkeypatch) -> None:
    created = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def invoke(self, prompt):
            created["prompt"] = prompt
            return SimpleNamespace(content=" Generated answer. ")

    monkeypatch.setattr(rag.providers, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(
        rag.providers,
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
        rag.providers,
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
    assert "base_url" not in created["kwargs"]
    assert "CTX=Context text" in created["prompt"]
    assert "Q=Question text" in created["prompt"]


def test_build_ollama_llm_passes_compose_host(monkeypatch) -> None:
    created = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def invoke(self, prompt):
            return SimpleNamespace(content="ok")

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setattr(rag.providers, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(
        rag.providers,
        "load_generation_config",
        lambda: {
            "model": "phi3:mini",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_new_tokens": 64,
            "do_sample": False,
            "fallback_to_dummy_on_error": False,
            "num_ctx": 2048,
            "timeout_seconds": 30,
        },
    )
    monkeypatch.setattr(rag.providers, "load_rag_prompt", lambda: "CTX={context}\nQ={question}")
    rag._generate_with_ollama("Context text", "Question text")
    assert created["kwargs"]["base_url"] == "http://ollama:11434"
    assert created["kwargs"]["timeout"] == 30


def test_generate_answer_falls_back_to_dummy_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_ollama",
        lambda context, query, settings=None: (_ for _ in ()).throw(
            ConnectionError("connection refused")
        ),
    )
    monkeypatch.setattr(
        rag.providers,
        "load_generation_config",
        lambda: {"fallback_to_dummy_on_error": True},
    )

    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=False,
    )

    assert "Ollama unavailable, falling back to dummy mode." in answer
    assert "no AI model is running here" in answer


def test_generate_answer_raises_structured_error_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_ollama",
        lambda context, query, settings=None: (_ for _ in ()).throw(
            RuntimeError("model not found")
        ),
    )
    monkeypatch.setattr(
        rag.providers,
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


def test_generate_answer_raises_structured_timeout_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_ollama",
        lambda context, query, settings=None: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(
        rag.providers,
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
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError on timeout when fallback is disabled.")


def test_resolve_llm_provider_name_env_wins_over_dummy_mode(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert rag.resolve_llm_provider_name(dummy_mode=True) == "ollama"

    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    assert rag.resolve_llm_provider_name(dummy_mode=False) == "dummy"


def test_resolve_llm_provider_name_maps_dummy_mode_when_env_unset() -> None:
    assert rag.resolve_llm_provider_name(dummy_mode=True) == "dummy"
    assert rag.resolve_llm_provider_name(dummy_mode=False) == "ollama"


def test_get_llm_provider_dummy_ollama_and_anthropic() -> None:
    assert isinstance(rag.get_llm_provider("dummy"), rag.DummyLLMProvider)
    assert isinstance(rag.get_llm_provider("OLLAMA"), rag.OllamaLLMProvider)
    assert isinstance(rag.get_llm_provider("anthropic"), rag.AnthropicLLMProvider)


def test_get_llm_provider_openai_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="openai"):
        rag.get_llm_provider("openai")


def test_get_llm_provider_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        rag.get_llm_provider("grok")


def test_generate_answer_llm_provider_dummy_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=False,
    )
    assert "no AI model is running here" in answer


def test_generate_answer_llm_provider_ollama_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_ollama",
        lambda context, query, settings=None: "via env",
    )
    answer = rag.generate_answer(
        context="Clause text",
        query="What applies?",
        dummy_mode=True,
    )
    assert answer == "via env"


def test_generate_answer_llm_provider_anthropic_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_anthropic",
        lambda context, query, settings=None: "via anthropic",
    )
    answer = rag.generate_answer(
        context="Clause text",
        query="What applies?",
        dummy_mode=True,
    )
    assert answer == "via anthropic"


def test_generate_with_anthropic_respects_config_and_prompt(monkeypatch) -> None:
    created = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def invoke(self, prompt):
            created["prompt"] = prompt
            return SimpleNamespace(content=" Cited answer. ")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag.providers, "ChatAnthropic", FakeChatAnthropic)
    monkeypatch.setattr(
        rag.providers,
        "load_generation_config",
        lambda: {
            "anthropic_model": "claude-haiku-4-5-20251001",
            "temperature": 0.55,
            "top_p": 0.9,
            "max_new_tokens": 128,
            "do_sample": False,
            "timeout_seconds": 30,
        },
    )
    monkeypatch.setattr(
        rag.providers,
        "load_rag_prompt",
        lambda: "CTX={context}\nQ={question}\nA=",
    )

    output = rag._generate_with_anthropic("Context text", "Question text")

    assert output == "Cited answer."
    assert created["kwargs"]["model"] == "claude-haiku-4-5-20251001"
    assert created["kwargs"]["api_key"] == "test-key-not-real"
    assert created["kwargs"]["max_tokens"] == 128
    assert created["kwargs"]["timeout"] == 30.0
    # Haiku 4.5 rejects temperature + top_p together; we send temperature only.
    assert "top_p" not in created["kwargs"]
    assert created["kwargs"]["temperature"] == 0.0
    # anthropic 2.x defaults to API 2026-10-21, which rejects this field.
    assert created["kwargs"]["default_headers"]["anthropic-version"] == "2023-06-01"
    assert "CTX=Context text" in created["prompt"]
    assert "Q=Question text" in created["prompt"]


def _sampling_temperature(payload: dict) -> float | None:
    """Temperature as sent: named arg on older SDKs, extra_body on anthropic>=1."""
    if "temperature" in payload:
        return payload["temperature"]
    extra = payload.get("extra_body") or {}
    return extra.get("temperature")


def _sampling_keys(payload: dict) -> set[str]:
    extra = payload.get("extra_body") or {}
    return {key for key in ("temperature", "top_p", "top_k") if key in payload or key in extra}


def test_haiku_temperature_stays_on_sampling_api_version(monkeypatch) -> None:
    """Reproduce the live 400 and require the request that Haiku 4.5 accepts.

    langchain-anthropic>=1.7 moves temperature into extra_body when the SDK
    dropped the named argument. anthropic 2.x then sends API version 2026-10-21,
    and that version returns ``temperature: Extra inputs are not permitted``
    for every model, including claude-haiku-4-5-20251001. The previous image
    stayed on anthropic 1.x (API 2023-06-01), which still honors temperature 0.
    """
    import anthropic
    import httpx2
    from anthropic.types import Message, TextBlock, Usage

    # Installed 1.x still reads this constant. Force the 2.x default so the
    # test fails if the builder stops pinning the header.
    monkeypatch.setattr(anthropic._client, "ANTHROPIC_API_VERSION", "2026-10-21")

    settings = {
        "temperature": 0.55,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "do_sample": False,
        "timeout_seconds": 30,
    }
    llm = rag.providers._build_anthropic_llm(settings, "test-key-not-real")
    captured: dict = {}

    class _Raw:
        def parse(self) -> Message:
            return Message(
                id="msg_test",
                type="message",
                content=[TextBlock(type="text", text="cited")],
                model=rag.DEFAULT_ANTHROPIC_MODEL,
                role="assistant",
                stop_reason="end_turn",
                usage=Usage(input_tokens=3, output_tokens=2),
            )

    def create(**kwargs):
        version = llm._client.default_headers.get("anthropic-version")
        captured["kwargs"] = kwargs
        captured["version"] = version
        temperature = _sampling_temperature(kwargs)
        if version != "2023-06-01" and temperature is not None:
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            body = {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "temperature: Extra inputs are not permitted",
                },
            }
            response = httpx2.Response(400, request=request, json=body)
            raise anthropic.BadRequestError(
                "temperature: Extra inputs are not permitted",
                response=response,
                body=body,
            )
        return _Raw()

    llm._client.messages.with_raw_response.create = create
    answer = llm.invoke("What is the notice period?")

    assert rag.DEFAULT_ANTHROPIC_MODEL == "claude-haiku-4-5-20251001"
    assert answer.content == "cited"
    assert captured["version"] == "2023-06-01"
    assert captured["kwargs"]["model"] == "claude-haiku-4-5-20251001"
    # do_sample false stays deterministic; top_p is not sent with temperature.
    assert _sampling_temperature(captured["kwargs"]) == 0.0
    assert _sampling_keys(captured["kwargs"]) == {"temperature"}


def test_generate_with_anthropic_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rag._generate_with_anthropic("context", "query")


def test_anthropic_provider_generate_routes_and_invokes(monkeypatch) -> None:
    captured = {}

    def _fake_generate(context: str, query: str, settings=None) -> str:
        captured["context"] = context
        captured["query"] = query
        captured["settings"] = settings
        return "anthropic answer"

    monkeypatch.setattr(rag.providers, "_generate_with_anthropic", _fake_generate)
    provider = rag.AnthropicLLMProvider(
        settings={
            "anthropic_model": "claude-haiku-4-5-20251001",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 64,
            "do_sample": False,
            "timeout_seconds": 30,
        }
    )
    answer = provider.generate("Section A: 30 days notice.", "Notice period?")
    assert answer == "anthropic answer"
    assert captured["query"] == "Notice period?"
    assert "30 days" in captured["context"]


def test_load_generation_config_anthropic_model_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    rag.load_generation_config.cache_clear()
    cfg = rag.load_generation_config()
    assert cfg["anthropic_model"] == rag.DEFAULT_ANTHROPIC_MODEL

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    rag.load_generation_config.cache_clear()
    cfg = rag.load_generation_config()
    assert cfg["anthropic_model"] == "claude-haiku-4-5"


def test_generate_answer_stream_dummy_yields_placeholder() -> None:
    chunks = list(
        rag.generate_answer_stream(
            context="Section A says notice period is 30 days.",
            query="What is the notice period?",
            dummy_mode=True,
        )
    )
    assert len(chunks) == 1
    assert "no AI model is running here" in chunks[0]
    assert "live pilot" in chunks[0]


def test_stub_llm_streams_canned_answer_token_by_token(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_STUB_LLM", "1")
    monkeypatch.delenv("DEMO_STUB_STREAM_DELAY_MS", raising=False)
    chunks = list(
        rag.generate_answer_stream(
            context="Section 4 - Term. Three years from signing.",
            query="How long does this agreement last?",
            dummy_mode=True,
        )
    )
    assert len(chunks) > 1
    assert "".join(chunks).startswith("3 years")


def test_stub_llm_stream_delay_pauses_between_tokens(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_STUB_LLM", "1")
    monkeypatch.setenv("DEMO_STUB_STREAM_DELAY_MS", "20")
    start = time.perf_counter()
    chunks = list(
        rag.generate_answer_stream(
            context="Section 4 - Term. Three years from signing.",
            query="How long does this agreement last?",
            dummy_mode=True,
        )
    )
    elapsed = time.perf_counter() - start
    assert len(chunks) > 1
    assert elapsed >= 0.02 * (len(chunks) - 1)


def test_generate_answer_empty_context_uses_refusal_copy() -> None:
    from ai_doc.rag.citations import INSUFFICIENT_CONTEXT_ANSWER, is_insufficient_answer

    answer = rag.generate_answer(
        context="   ",
        query="Does this specify liquidated damages?",
        dummy_mode=True,
    )
    assert answer == INSUFFICIENT_CONTEXT_ANSWER
    assert is_insufficient_answer(answer)


def test_generate_answer_stream_empty_query_yields_error() -> None:
    chunks = list(
        rag.generate_answer_stream(
            context="Some context",
            query="   ",
            dummy_mode=True,
        )
    )
    assert chunks == ["Please enter a non-empty question."]


def test_dummy_provider_stream_yields_one_chunk() -> None:
    provider = rag.DummyLLMProvider()
    chunks = list(provider.stream("ctx", "q"))
    assert chunks == [rag.DUMMY_UI_RESPONSE]


def test_stream_with_anthropic_uses_langchain_stream(monkeypatch) -> None:
    created = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def stream(self, prompt):
            created["prompt"] = prompt
            yield SimpleNamespace(content="Cited ")
            yield SimpleNamespace(content="answer.")

        def invoke(self, prompt):
            raise AssertionError("invoke should not be used for streaming path")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rag.providers, "ChatAnthropic", FakeChatAnthropic)
    monkeypatch.setattr(
        rag.providers,
        "load_generation_config",
        lambda: {
            "anthropic_model": "claude-haiku-4-5-20251001",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 64,
            "do_sample": False,
            "timeout_seconds": 30,
        },
    )
    monkeypatch.setattr(rag.providers, "load_rag_prompt", lambda: "CTX={context}\nQ={question}\nA=")

    chunks = list(rag._stream_with_anthropic("Context text", "Question text"))

    assert chunks == ["Cited ", "answer."]
    assert created["kwargs"]["model"] == "claude-haiku-4-5-20251001"
    assert created["kwargs"]["api_key"] == "test-key-not-real"
    assert "CTX=Context text" in created["prompt"]
    assert "Q=Question text" in created["prompt"]


def test_generate_answer_stream_anthropic_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def _fake_stream(context: str, query: str, settings=None):
        yield "chunk-a"
        yield "chunk-b"

    monkeypatch.setattr(rag.providers, "_stream_with_anthropic", _fake_stream)

    chunks = list(
        rag.generate_answer_stream(
            context="Clause text",
            query="What applies?",
            dummy_mode=True,
        )
    )
    assert chunks == ["chunk-a", "chunk-b"]


def test_anthropic_provider_stream_falls_back_to_generate(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def _failing_stream(context: str, query: str, settings=None):
        raise ConnectionError("stream broken")
        yield  # pragma: no cover — make this a generator

    monkeypatch.setattr(rag.providers, "_stream_with_anthropic", _failing_stream)
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_anthropic",
        lambda context, query, settings=None: "fallback full answer",
    )

    provider = rag.AnthropicLLMProvider(
        settings={
            "anthropic_model": "claude-haiku-4-5-20251001",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_new_tokens": 64,
            "do_sample": False,
            "timeout_seconds": 30,
        }
    )
    chunks = list(provider.stream("Section A: 30 days.", "Notice?"))
    assert chunks == ["fallback full answer"]


def test_stream_with_ollama_uses_langchain_stream(monkeypatch) -> None:
    created = {}

    class FakeChatOllama:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def stream(self, prompt):
            created["prompt"] = prompt
            yield SimpleNamespace(content="Hello ")
            yield SimpleNamespace(content="world")

        def invoke(self, prompt):
            raise AssertionError("invoke should not be used for streaming path")

    monkeypatch.setattr(rag.providers, "ChatOllama", FakeChatOllama)
    monkeypatch.setattr(
        rag.providers,
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
    monkeypatch.setattr(rag.providers, "load_rag_prompt", lambda: "CTX={context}\nQ={question}\nA=")

    chunks = list(rag._stream_with_ollama("Context text", "Question text"))

    assert chunks == ["Hello ", "world"]
    assert created["kwargs"]["model"] == "phi3:mini"
    assert created["kwargs"]["temperature"] == 0.0
    assert "CTX=Context text" in created["prompt"]


def test_generate_answer_stream_ollama_falls_back_to_generate(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    def _failing_stream(context: str, query: str, settings=None):
        raise TimeoutError("stream timed out")
        yield  # pragma: no cover

    monkeypatch.setattr(rag.providers, "_stream_with_ollama", _failing_stream)
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_ollama",
        lambda context, query, settings=None: "non-stream answer",
    )

    chunks = list(
        rag.generate_answer_stream(
            context="Clause text",
            query="What applies?",
            dummy_mode=False,
        )
    )
    assert chunks == ["non-stream answer"]
