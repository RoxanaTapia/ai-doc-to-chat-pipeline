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
    """Keep provider selection tests independent of the host env. No live API calls."""
    for name in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "XAI_API_KEY",
        "XAI_MODEL",
        "DEMO_STUB_LLM",
        "USE_DUMMY_GENERATOR",
    ):
        monkeypatch.delenv(name, raising=False)
    rag.load_generation_config.cache_clear()


_SAMPLE_SETTINGS = {
    "anthropic_model": "claude-haiku-4-5-20251001",
    "openai_model": "gpt-4.1-mini",
    "xai_model": "grok-4.7",
    "temperature": 0.55,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "do_sample": False,
    "timeout_seconds": 30,
}


def test_generate_answer_dummy_mode_returns_placeholder() -> None:
    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=True,
    )
    assert "no AI model is running here" in answer
    assert "live pilot" in answer


def test_generate_answer_real_mode_uses_anthropic_by_default(monkeypatch) -> None:
    captured = {}

    def _fake_generate(context: str, query: str, settings=None) -> str:
        captured["context"] = context
        captured["query"] = query
        captured["settings"] = settings
        return "final from anthropic"

    monkeypatch.setattr(rag.providers, "_generate_with_anthropic", _fake_generate)

    answer = rag.generate_answer(
        context="Clause 5: Non-compete applies for 3 years.",
        query="Summarize restrictions.",
        dummy_mode=False,
    )

    assert answer == "final from anthropic"
    assert captured["query"] == "Summarize restrictions."
    assert "Clause 5" in captured["context"]
    assert isinstance(captured["settings"], dict)


def test_resolve_llm_provider_name_env_wins_over_dummy_mode(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert rag.resolve_llm_provider_name(dummy_mode=True) == "openai"

    monkeypatch.setenv("LLM_PROVIDER", "dummy")
    assert rag.resolve_llm_provider_name(dummy_mode=False) == "dummy"

    monkeypatch.setenv("LLM_PROVIDER", "  XAI  ")
    assert rag.resolve_llm_provider_name(dummy_mode=True) == "xai"


def test_resolve_llm_provider_name_maps_dummy_mode_when_env_unset() -> None:
    assert rag.resolve_llm_provider_name(dummy_mode=True) == "dummy"
    assert rag.resolve_llm_provider_name(dummy_mode=False) == "anthropic"
    assert rag.DEFAULT_LLM_PROVIDER == "anthropic"


@pytest.mark.parametrize(
    ("env_value", "provider_type"),
    [
        ("anthropic", rag.AnthropicLLMProvider),
        ("openai", rag.OpenAILLMProvider),
        ("xai", rag.XAILLMProvider),
        ("ANTHROPIC", rag.AnthropicLLMProvider),
    ],
)
def test_llm_provider_env_selects_each_backend(monkeypatch, env_value, provider_type) -> None:
    monkeypatch.setenv("LLM_PROVIDER", env_value)
    name = rag.resolve_llm_provider_name(dummy_mode=True)
    assert isinstance(rag.get_llm_provider(name), provider_type)


def test_get_llm_provider_dummy_and_hosted() -> None:
    assert isinstance(rag.get_llm_provider("dummy"), rag.DummyLLMProvider)
    assert isinstance(rag.get_llm_provider("anthropic"), rag.AnthropicLLMProvider)
    assert isinstance(rag.get_llm_provider("openai"), rag.OpenAILLMProvider)
    assert isinstance(rag.get_llm_provider("xai"), rag.XAILLMProvider)


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


@pytest.mark.parametrize(
    ("provider", "key_env", "wrapper"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "_generate_with_anthropic"),
        ("openai", "OPENAI_API_KEY", "_generate_with_openai"),
        ("xai", "XAI_API_KEY", "_generate_with_xai"),
    ],
)
def test_selected_provider_missing_key_is_a_clear_error(
    monkeypatch, provider, key_env, wrapper
) -> None:
    """The error names the missing key and never builds a client."""

    def _refuse_client(*args, **kwargs):
        raise AssertionError("a missing key must not construct a model client")

    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setattr(rag.providers, "ChatAnthropic", _refuse_client)
    monkeypatch.setattr(rag.providers, "ChatOpenAI", _refuse_client)
    with pytest.raises(RuntimeError, match=key_env) as exc_info:
        rag.generate_answer(
            context="Section A says notice period is 30 days.",
            query="What is the notice period?",
            dummy_mode=True,
        )
    message = str(exc_info.value)
    assert f"LLM_PROVIDER={provider}" in message
    assert "not set" in message
    assert wrapper.startswith("_generate_with_")


def test_dummy_mode_answers_without_any_api_key(monkeypatch) -> None:
    def _refuse_client(*args, **kwargs):
        raise AssertionError("dummy mode must not construct a model client")

    monkeypatch.setattr(rag.providers, "ChatAnthropic", _refuse_client)
    monkeypatch.setattr(rag.providers, "ChatOpenAI", _refuse_client)
    answer = rag.generate_answer(
        context="Section A says notice period is 30 days.",
        query="What is the notice period?",
        dummy_mode=True,
    )
    assert "no AI model is running here" in answer
    assert "live pilot" in answer


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


def test_load_generation_config_model_defaults_and_env(monkeypatch) -> None:
    rag.load_generation_config.cache_clear()
    cfg = rag.load_generation_config()
    assert cfg["provider"] == "anthropic"
    assert cfg["anthropic_model"] == rag.DEFAULT_ANTHROPIC_MODEL
    assert cfg["openai_model"] == rag.DEFAULT_OPENAI_MODEL
    assert cfg["xai_model"] == rag.DEFAULT_XAI_MODEL

    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("XAI_MODEL", "grok-4")
    rag.load_generation_config.cache_clear()
    cfg = rag.load_generation_config()
    assert cfg["anthropic_model"] == "claude-haiku-4-5"
    assert cfg["openai_model"] == "gpt-4.1"
    assert cfg["xai_model"] == "grok-4"


def test_openai_and_xai_builders_use_fakes_and_their_own_keys(monkeypatch) -> None:
    created = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def invoke(self, prompt):
            created[-1]["prompt"] = prompt
            return SimpleNamespace(content=" Hosted answer. ")

    monkeypatch.setattr(rag.providers, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(rag.providers, "load_rag_prompt", lambda: "CTX={context}\nQ={question}\nA=")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    openai_answer = rag._generate_with_openai(
        "Context text", "Question text", settings=_SAMPLE_SETTINGS
    )
    xai_answer = rag._generate_with_xai(
        "Context text", "Question text", settings=_SAMPLE_SETTINGS
    )

    assert openai_answer == "Hosted answer."
    assert xai_answer == "Hosted answer."
    assert created[0]["model"] == "gpt-4.1-mini"
    assert created[0]["api_key"] == "openai-test-key"
    assert created[0]["max_tokens"] == 128
    assert created[0]["temperature"] == 0.0
    assert "base_url" not in created[0]
    assert "top_p" not in created[0]
    assert created[1]["model"] == "grok-4.7"
    assert created[1]["api_key"] == "xai-test-key"
    assert created[1]["base_url"] == rag.XAI_BASE_URL
    assert "CTX=Context text" in created[0]["prompt"]
    assert "Q=Question text" in created[1]["prompt"]


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


def test_generate_answer_stream_openai_falls_back_to_generate(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    def _failing_stream(context: str, query: str, settings=None):
        raise TimeoutError("stream timed out")
        yield  # pragma: no cover

    monkeypatch.setattr(rag.providers, "_stream_with_openai", _failing_stream)
    monkeypatch.setattr(
        rag.providers,
        "_generate_with_openai",
        lambda context, query, settings=None: "non-stream answer",
    )

    chunks = list(
        rag.generate_answer_stream(
            context="Clause text",
            query="What applies?",
            dummy_mode=True,
        )
    )
    assert chunks == ["non-stream answer"]
