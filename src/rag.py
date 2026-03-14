import re
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
APP_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = APP_ROOT / "configs" / "prompts.yaml"
DEFAULT_OLLAMA_MODEL = "phi3:mini"
OLLAMA_PROMPT_TEMPLATE = (
    "Answer concisely using only this context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\n"
    "Answer:"
)


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


def _generate_with_ollama(context: str, query: str) -> str:
    """Milestone-4-ready local generator skeleton."""
    llm = ChatOllama(
        model=DEFAULT_OLLAMA_MODEL,
        temperature=0.3,
        num_ctx=4096,
        timeout=90,
    )
    prompt = OLLAMA_PROMPT_TEMPLATE.format(context=context, query=query)
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, "content") else str(response).strip()


def generate_answer(context: str, query: str, dummy_mode: bool = True) -> str:
    """Generate a response with dummy mode defaulted for issue #20."""
    if not (context or "").strip():
        return "I could not find relevant information in the document to answer this question."

    safe_query = _normalize_untrusted_text(query, max_chars=2000)
    safe_context = _normalize_untrusted_text(context, max_chars=12000)

    if dummy_mode:
        return (
            "Dummy answer:\n"
            f"{safe_query}\n\n"
            "Based on the document:\n"
            f"{safe_context[:400]}..."
        )

    # Milestone 4 placeholder branch; keep real Ollama skeleton above for next step.
    return f"Ollama not ready yet.\nFallback: {safe_query} -> {safe_context[:200]}..."
