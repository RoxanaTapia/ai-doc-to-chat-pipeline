"""Process-wide sample NDA index.

The container entrypoint builds this before Streamlit listens, in the same
process, so the first question does not load the embedding model or the index.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import yaml
from langchain_community.vectorstores import FAISS

from ai_doc.rag.chunking import chunk_pages
from ai_doc.rag.config import CONFIG_PATH
from ai_doc.rag.ingestion import extract_pdf

logger = logging.getLogger(__name__)

SAMPLE_NDA_NAME = "sample-nda.pdf"
_EMBEDDINGS: dict[tuple, object] = {}
_INDEXES: dict[tuple, dict] = {}


def sample_nda_path() -> Path:
    """Repo path to the walkthrough NDA PDF."""
    # apps/demo/sample_warm.py → repo root.
    return Path(__file__).resolve().parents[2] / "docs" / "product" / SAMPLE_NDA_NAME


def _use_stub_embeddings() -> bool:
    raw = os.getenv("USE_STUB_EMBEDDINGS")
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _index_key(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    header_split: bool,
) -> tuple:
    return (
        model_name,
        int(chunk_size),
        int(chunk_overlap),
        bool(header_split),
        _use_stub_embeddings(),
    )


def index_settings() -> tuple[str, int, int, bool]:
    """Chunking and embedding settings the demo uses for the sample NDA."""
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    chunking = config["chunking"]
    return (
        str(config["embeddings"]["model_name"]),
        int(chunking["chunk_size"]),
        int(chunking["chunk_overlap"]),
        bool(chunking.get("split_on_legal_headers", True)),
    )


def embeddings_for(model_name: str):
    """Return the process-wide embedding model, loading it on first use."""
    key = (model_name, _use_stub_embeddings())
    cached = _EMBEDDINGS.get(key)
    if cached is not None:
        return cached
    if _use_stub_embeddings():
        from langchain_core.embeddings import FakeEmbeddings

        model = FakeEmbeddings(size=32)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(model_name=model_name)
    _EMBEDDINGS[key] = model
    return model


def _page_text_counts(page_docs) -> tuple[int, int]:
    # Same helper the upload path uses. Imported lazily to avoid a cycle.
    from session_upload import page_text_counts

    return page_text_counts(page_docs)


def _build_sample_index(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    header_split: bool,
) -> dict:
    path = sample_nda_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_bytes = path.read_bytes()
    extracted = extract_pdf(path, enable_ocr=False)
    chunks = chunk_pages(
        extracted.page_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_on_legal_headers=header_split,
    )
    if not chunks:
        raise ValueError(f"No chunks in {path.name}")
    store = FAISS.from_documents(chunks, embeddings_for(model_name))
    text_pages, blank_pages = _page_text_counts(extracted.page_docs)
    return {
        "bytes": file_bytes,
        "chunks": chunks,
        "vector_store": store,
        "pages": len(extracted.page_docs),
        "text_pages": text_pages,
        "blank_pages": blank_pages,
        "chars": len(extracted.preview_text.strip()),
        "hash": hashlib.sha256(file_bytes).hexdigest(),
    }


def sample_index_for(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    header_split: bool,
) -> dict:
    """Return the sample NDA index, building it once per process and settings."""
    key = _index_key(model_name, chunk_size, chunk_overlap, header_split)
    cached = _INDEXES.get(key)
    if cached is not None:
        return cached
    bundle = _build_sample_index(model_name, chunk_size, chunk_overlap, header_split)
    _INDEXES[key] = bundle
    return bundle


def warm_sample_index() -> dict | None:
    """Build the sample index now. A failure is logged so the server can still start."""
    try:
        model_name, chunk_size, chunk_overlap, header_split = index_settings()
        return sample_index_for(model_name, chunk_size, chunk_overlap, header_split)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        yaml.YAMLError,
        KeyError,
        TypeError,
    ):
        logger.exception("Sample NDA index was not warmed. The first session will load it.")
        return None
