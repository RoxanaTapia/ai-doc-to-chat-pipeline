from pathlib import Path
import sys

from langchain.prompts import PromptTemplate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import load_rag_prompt  # noqa: E402


def _build_prompt() -> PromptTemplate:
    template = load_rag_prompt()
    return PromptTemplate(template=template, input_variables=["context", "question"])


def test_rag_prompt_renders_with_context_and_question() -> None:
    prompt = _build_prompt()

    rendered = prompt.format(
        context="Chunk 1: Non-compete clause...\nChunk 2: Term and scope...",
        question="Does this document contain a non-compete clause?",
    )

    assert "Extracted context:" in rendered
    assert "Chunk 1: Non-compete clause..." in rendered
    assert "Chunk 2: Term and scope..." in rendered
    assert "User question:" in rendered
    assert "Does this document contain a non-compete clause?" in rendered


def test_rag_prompt_handles_empty_context() -> None:
    prompt = _build_prompt()

    rendered = prompt.format(context="", question="Test question?")

    assert "Extracted context:" in rendered
    assert "User question:" in rendered
    assert "Test question?" in rendered


def test_rag_prompt_variables_match_expected_contract() -> None:
    prompt = _build_prompt()

    assert set(prompt.input_variables) == {"context", "question"}


def test_rag_prompt_preserves_multiline_context() -> None:
    prompt = _build_prompt()
    multiline_context = (
        "Section 5. Non-compete. Party shall not compete for 3 years.\n"
        "Section 6. Territory. Applies across all EU member states.\n"
        "Section 7. Exceptions. Internal advisory work is allowed."
    )

    rendered = prompt.format(context=multiline_context, question="Summarize restrictions.")

    assert "3 years" in rendered
    assert "EU member states" in rendered
    assert "Summarize restrictions." in rendered
