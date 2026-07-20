"""RAG generation package: config loaders, LLM providers, and answer entrypoints."""

from rag.config import (
    APP_ROOT,
    CONFIG_PATH,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    PROMPTS_PATH,
    load_generation_config,
    load_rag_prompt,
)
from rag.generation import generate_answer, generate_answer_stream
from rag.providers import (
    DUMMY_UI_RESPONSE,
    SUPPORTED_LLM_PROVIDERS,
    AnthropicLLMProvider,
    ChatAnthropic,
    ChatOllama,
    DummyLLMProvider,
    LLMProvider,
    OllamaLLMProvider,
    _generate_with_anthropic,
    _generate_with_ollama,
    _stream_with_anthropic,
    _stream_with_ollama,
    get_llm_provider,
    resolve_llm_provider_name,
)

__all__ = [
    "APP_ROOT",
    "CONFIG_PATH",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_PROMPT_TEMPLATE",
    "DUMMY_UI_RESPONSE",
    "LLMProvider",
    "PROMPTS_PATH",
    "SUPPORTED_LLM_PROVIDERS",
    "AnthropicLLMProvider",
    "ChatAnthropic",
    "ChatOllama",
    "DummyLLMProvider",
    "OllamaLLMProvider",
    "generate_answer",
    "generate_answer_stream",
    "get_llm_provider",
    "load_generation_config",
    "load_rag_prompt",
    "resolve_llm_provider_name",
    "_generate_with_anthropic",
    "_generate_with_ollama",
    "_stream_with_anthropic",
    "_stream_with_ollama",
]
