from pathlib import Path
import sys

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.citations import assemble_context, readable_excerpt  # noqa: E402
from rag.retrieval import normalize_scores  # noqa: E402


def test_normalize_scores_scales_to_unit_interval() -> None:
    assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_assemble_context_adds_page_separators() -> None:
    docs = [
        (Document(page_content="Alpha", metadata={"page": 1}), 0.9),
        (Document(page_content="Beta", metadata={"page": 2}), 0.8),
    ]
    text = assemble_context(docs, use_page_separators=True)
    assert "─── Page 1 ───" in text
    assert "Page 2: Beta" in text


def test_readable_excerpt_cuts_on_sentence() -> None:
    text = "First sentence. Second sentence continues for a while."
    excerpt = readable_excerpt(text, 20)
    assert excerpt.endswith("…")
    assert "First" in excerpt
