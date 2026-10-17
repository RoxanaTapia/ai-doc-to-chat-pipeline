import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import load_rag_prompt  # noqa: E402
from rag.citations import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402


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


def test_rag_prompt_includes_reliability_rules() -> None:
    template = load_rag_prompt()

    assert "Never invent section numbers" in template
    assert "must not do" in template
    assert "I agree not to" in template
    assert INSUFFICIENT_CONTEXT_ANSWER in template
    assert "partial or adjacent" in template
    assert "injunctive relief" in template
    near_miss = "no fee is specified; Section 6 allows injunctive relief"
    assert near_miss in template
    assert "Use the exact sentinel sentence instead" in template
    assert "I could not find enough information in the document to answer." not in template


def test_streamlit_disables_browser_usage_stats() -> None:
    text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in text


def test_sample_nda_pdf_is_present() -> None:
    pdf = PROJECT_ROOT / "docs" / "product" / "sample-nda.pdf"
    assert pdf.is_file()
    assert pdf.stat().st_size > 1000


def test_sample_nda_pdf_keeps_storyboard_facts() -> None:
    import pymupdf

    doc = pymupdf.open(PROJECT_ROOT / "docs" / "product" / "sample-nda.pdf")
    pages = [page.get_text() for page in doc]
    full = "\n".join(pages)
    assert len(pages) >= 3
    assert "Convert this file to PDF" not in full
    assert "penalty" not in full.lower()
    assert "three (3) years" in pages[1]
    assert "Section 4" in pages[1]
    assert "Spain" in pages[2]
    assert "Madrid" in pages[2]
    assert "CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT" in pages[0]
    title_lines = [
        line
        for block in doc[0].get_text("blocks")
        for line in str(block[4]).splitlines()
        if "CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT" in line
    ]
    assert title_lines, "title should appear as one line on page 1"
    remedies_pages = [text for text in pages if "Remedies" in text]
    assert remedies_pages
    assert all("irreparable" in text for text in remedies_pages)


def test_starter_chips_include_two_hits_and_one_miss() -> None:
    from page_copy import STARTER_CHIPS

    questions = [chip["question"] for chip in STARTER_CHIPS]
    assert questions == [
        "How long does this agreement last?",
        "Which country's law applies, and where are disputes settled?",
        "What is the penalty fee if Beta Labs breaks this agreement?",
    ]
    assert [chip["label"] for chip in STARTER_CHIPS] == questions


def test_privacy_line_follows_generator_without_overclaiming() -> None:
    from page_copy import _CLIENT_COPY, privacy_line_for_provider

    ollama = privacy_line_for_provider("ollama")
    assert ollama == (
        "Runs on this server: your PDF, search and answers never leave it. "
        "Files are deleted right after processing."
    )
    assert "passages" not in ollama
    assert "Anthropic" not in ollama

    anthropic = privacy_line_for_provider("anthropic")
    assert anthropic == (
        "Your PDF and search stay on this server; only the few passages "
        "needed for an answer are sent to a hosted model (Anthropic). "
        "Files are deleted right after processing."
    )
    assert "never leave" not in anthropic

    dummy = privacy_line_for_provider("dummy")
    assert dummy == _CLIENT_COPY["privacy_memory"]
    assert dummy == "Files are deleted right after processing."
    assert "never leave" not in dummy
    assert "passages" not in dummy
    assert privacy_line_for_provider("openai") == _CLIENT_COPY["privacy_memory"]
    assert "never stored" not in dummy
    assert "infrastructure you control" not in _CLIENT_COPY["hero_lead"]
