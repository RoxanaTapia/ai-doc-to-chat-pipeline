"""Retrieval core: grounded answers from a PDF, with page citations.

The Streamlit demo stays outside this package. Importing ``ai_doc`` does not
import Streamlit.
"""

from ai_doc.api.app import app
from ai_doc.rag import (
    generate_answer,
    generate_answer_stream,
    resolve_llm_provider_name,
)

__version__ = "1.0.0"

__all__ = [
    "app",
    "generate_answer",
    "generate_answer_stream",
    "resolve_llm_provider_name",
]
