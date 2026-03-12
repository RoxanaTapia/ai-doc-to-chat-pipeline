import re
from functools import lru_cache
from pathlib import Path

import yaml
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
APP_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"


def _normalize_untrusted_text(text: str, max_chars: int) -> str:
    """Trim and sanitize untrusted user/document text for prompting."""
    cleaned = CONTROL_CHARS_RE.sub("", text or "").strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars]
    return cleaned


@lru_cache(maxsize=1)
def load_rag_prompt() -> str:
    """Load and cache the raw RAG prompt template from YAML config."""
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config["rag_prompt"]["template"]


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer with local Ollama.
    Uses a lightweight model and a safe fallback while setup is in progress.
    """
    if not (context or "").strip():
        return "I could not find relevant information in the document to answer."

    # Treat both inputs as untrusted data (prompt-injection can come from either).
    safe_query = _normalize_untrusted_text(query, max_chars=2000)
    safe_context = _normalize_untrusted_text(context, max_chars=12000)
    template = load_rag_prompt()
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=safe_context, question=safe_query)

    try:
        # phi3:mini is a practical default for older Intel Macs.
        llm = ChatOllama(
            model="phi3:mini",
            temperature=0.0,
            num_ctx=4096,
            timeout=120,
        )
        response = llm.invoke(formatted_prompt)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as exc:
        estimated_words = max(1, len(safe_context.split()) // 15)
        return (
            "Simulated answer (Ollama setup in progress): "
            f"Based on the retrieved context, the relevant section appears to contain about {estimated_words} "
            f"key words of evidence. Error: {exc}. Please try again and verify Ollama is running."
        )
