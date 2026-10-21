"""Session keys, upload cache, sample NDA, and the single document status line."""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

import streamlit as st

from page_copy import _CLIENT_COPY, sample_preload_error_copy, should_preload_sample

# Stable for the lifetime of the Streamlit process (survives reruns, changes on container restart).
_APP_BOOT_ID = str(os.getpid())
SAMPLE_NDA_NAME = "sample-nda.pdf"
_STATUS_KINDS = frozenset({"empty", "reading", "ready", "answering", "error"})


def sample_nda_path() -> Path:
    """Repo path to the walkthrough NDA PDF."""
    return Path(__file__).resolve().parent.parent / "docs" / "product" / SAMPLE_NDA_NAME


def _on_new_browser_session() -> None:
    """Drop a stale upload when the browser session is new, so a ghost name is not ready."""
    if st.session_state.get("_app_boot_id") == _APP_BOOT_ID:
        return
    st.session_state._app_boot_id = _APP_BOOT_ID
    st.session_state.uploader_key_version = int(st.session_state.get("uploader_key_version", 0)) + 1
    for key in ("uploaded_pdf_bytes", "uploaded_pdf_name", "uploaded_pdf_hash"):
        st.session_state.pop(key, None)
    st.session_state.last_processed_name = None
    st.session_state.last_processed_hash = None
    st.session_state.current_file = None
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.indexed_doc_stats = None
    st.session_state.bm25_state = None
    st.session_state.own_pdf_mode = False
    _set_doc_status("empty", _CLIENT_COPY["doc_empty"])


def _upload_in_flight(*, uploaded_file, resolved_upload: tuple[bytes, str] | None) -> bool:
    """True when we have file bytes but no searchable index yet."""
    if _document_is_indexed():
        return False
    return (
        resolved_upload is not None
        or uploaded_file is not None
        or bool(st.session_state.get("uploaded_pdf_bytes"))
    )


def page_text_counts(page_docs) -> tuple[int, int]:
    """Return (pages with text, pages with none) from extracted page documents."""
    text_pages = 0
    blank_pages = 0
    for doc in page_docs or []:
        if (getattr(doc, "page_content", None) or "").strip():
            text_pages += 1
        else:
            blank_pages += 1
    return text_pages, blank_pages


def _set_indexed_doc_stats(
    *,
    file_name: str,
    pages: int,
    chars: int,
    reused_index: bool,
    header_split: bool,
    chunks: int | None = None,
    text_pages: int | None = None,
    blank_pages: int | None = None,
) -> None:
    """Remember page counts for the ready line and the refusal footnote."""
    st.session_state.indexed_doc_stats = {
        "file_name": file_name,
        "pages": pages,
        "chars": chars,
        "reused_index": reused_index,
        "chunks": chunks,
        "header_split": header_split,
        "text_pages": pages if text_pages is None else text_pages,
        "blank_pages": 0 if blank_pages is None else blank_pages,
    }


def _set_dev_index_logs(
    *,
    chunks_msg: str | None = None,
    faiss_msg: str | None = None,
) -> None:
    """Keep indexing notes in session so they render above chat, not mid-turn."""
    logs = dict(st.session_state.get("dev_index_logs") or {})
    if chunks_msg is not None:
        logs["chunks"] = chunks_msg
    if faiss_msg is not None:
        logs["faiss"] = faiss_msg
    st.session_state.dev_index_logs = logs


def _clear_dev_index_logs() -> None:
    """Drop indexing notes when the document is cleared."""
    st.session_state.pop("dev_index_logs", None)


def _clear_cached_upload() -> None:
    """Drop persisted upload bytes when the document is cleared."""
    for key in ("uploaded_pdf_bytes", "uploaded_pdf_name", "uploaded_pdf_hash"):
        st.session_state.pop(key, None)


def _cache_upload(file_bytes: bytes, file_name: str) -> str:
    """Keep the upload in session so a rerun does not lose the file buffer."""
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    st.session_state.uploaded_pdf_bytes = file_bytes
    st.session_state.uploaded_pdf_name = file_name
    st.session_state.uploaded_pdf_hash = file_hash
    return file_hash


def mark_sample_preload_failed(reason: str | None = None) -> None:
    """Stop the first-paint spinner and remember a short retry banner."""
    copy = sample_preload_error_copy()
    _clear_cached_upload()
    st.session_state.sample_preload_failed = True
    _set_doc_status("error", reason or copy["error"])


def mark_upload_unreadable() -> None:
    """Drop a bad upload and leave Ask off until a valid document is ready.

    Clears the leftover index so answers cannot keep coming from the sample
    while an upload error is on screen. Offers the same Use sample NDA retry
    as a failed first-paint preload.
    """
    message = _CLIENT_COPY["pdf_unreadable"]
    _reset_document_state(bump_uploader_key=True)
    st.session_state.own_pdf_mode = True
    st.session_state.show_own_pdf = True
    st.session_state.sample_preload_failed = True
    st.session_state._uploader_widget_had_file = False
    _set_doc_status("error", message)


def retry_sample_nda() -> bool:
    """Clear the preload error and try the walkthrough NDA again."""
    st.session_state.sample_preload_failed = False
    return load_sample_nda()


def render_sample_preload_error() -> None:
    """Short error plus a Use sample NDA retry when first-paint indexing fails."""
    if not st.session_state.get("sample_preload_failed"):
        return
    copy = sample_preload_error_copy()
    detail = st.session_state.get("doc_status_detail") or copy["error"]
    st.error(detail)
    if st.button(copy["retry"], key="retry_sample_nda"):
        if retry_sample_nda():
            st.rerun()


def preload_sample_nda() -> bool:
    """Load the walkthrough NDA on first paint when the visitor has not uploaded."""
    if st.session_state.get("sample_preload_failed"):
        return False
    own_pdf_mode = bool(st.session_state.get("own_pdf_mode"))
    has_upload = bool(st.session_state.get("uploaded_pdf_bytes")) or bool(
        st.session_state.get("current_file")
    )
    if not should_preload_sample(
        own_pdf_mode=own_pdf_mode,
        has_upload=has_upload,
        is_indexed=_document_is_indexed(),
    ):
        return False
    return load_sample_nda()


def load_sample_nda() -> bool:
    """Cache the walkthrough NDA so indexing starts without a widget upload."""
    path = sample_nda_path()
    if not path.is_file():
        _set_doc_status("error", _CLIENT_COPY["sample_missing"])
        return False
    _reset_document_state(bump_uploader_key=True)
    st.session_state.own_pdf_mode = False
    _cache_upload(path.read_bytes(), SAMPLE_NDA_NAME)
    _set_doc_status(
        "reading",
        _CLIENT_COPY["doc_reading"].format(
            name=SAMPLE_NDA_NAME,
            step=_CLIENT_COPY["progress_prepare"],
        ),
    )
    return True


def _resolve_upload(uploaded_file) -> tuple[bytes, str] | None:
    """Return file bytes and name from the widget or the session cache."""
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    cached_name = st.session_state.get("uploaded_pdf_name")
    cached_bytes = st.session_state.get("uploaded_pdf_bytes")
    if cached_name and cached_bytes:
        return cached_bytes, cached_name
    return None


def _document_is_indexed() -> bool:
    """True once this session has a searchable index."""
    return st.session_state.vector_store is not None


def _chat_input_placeholder(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
) -> str:
    """Tell the visitor whether they can ask, or still need to wait."""
    if _document_is_indexed():
        return _CLIENT_COPY["chat_ready"]
    if _upload_in_flight(uploaded_file=uploaded_file, resolved_upload=resolved_upload):
        return _CLIENT_COPY["chat_indexing"]
    return _CLIENT_COPY["chat_waiting"]


def _chat_blocked_user_message(uploaded_file) -> str:
    """Explain why a question cannot run yet."""
    if _document_is_indexed():
        return ""
    if uploaded_file is not None or st.session_state.get("uploaded_pdf_bytes"):
        return _CLIENT_COPY["chat_blocked_indexing"]
    return _CLIENT_COPY["chat_blocked_empty"]


def _set_doc_status(kind: str, message: str) -> None:
    """Remember the single status line for this session."""
    st.session_state.doc_status = kind if kind in _STATUS_KINDS else "empty"
    st.session_state.doc_status_detail = message


def _reading_message(name: str, step: str) -> str:
    return _CLIENT_COPY["doc_reading"].format(name=name, step=step)


def _ready_message(name: str, pages: int | None = None) -> str:
    page_count = pages
    if page_count is None:
        stats = st.session_state.get("indexed_doc_stats") or {}
        page_count = int(stats.get("pages") or 0)
    if name == SAMPLE_NDA_NAME:
        if page_count:
            return _CLIENT_COPY["doc_ready"].format(pages=page_count)
        return "Sample NDA"
    if page_count:
        return _CLIENT_COPY["doc_ready_named"].format(name=name, pages=page_count)
    return name


def _derive_document_status(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
) -> tuple[str, str]:
    """Pick empty / reading / ready / answering / error from session + upload."""
    if st.session_state.get("doc_status") == "answering":
        return "answering", _CLIENT_COPY["doc_answering"]

    err = st.session_state.get("last_generation_error") or {}
    if err.get("user_reason"):
        return "error", str(err["user_reason"])
    if st.session_state.get("doc_status") == "error" and st.session_state.get("doc_status_detail"):
        return "error", st.session_state.doc_status_detail

    stats = st.session_state.get("indexed_doc_stats") or {}
    if _document_is_indexed():
        name = stats.get("file_name") or st.session_state.current_file or "Document"
        return "ready", _ready_message(name)

    if _upload_in_flight(uploaded_file=uploaded_file, resolved_upload=resolved_upload):
        name = (
            st.session_state.get("uploaded_pdf_name")
            or (resolved_upload[1] if resolved_upload else None)
            or (getattr(uploaded_file, "name", None))
            or "document"
        )
        step = st.session_state.get("doc_status_detail")
        if st.session_state.get("doc_status") == "reading" and step:
            return "reading", step
        return "reading", _reading_message(name, _CLIENT_COPY["progress_prepare"])

    return "empty", _CLIENT_COPY["doc_empty"]


def paint_status(slot, kind: str, message: str) -> None:
    """Draw the one status pill and remember it."""
    _set_doc_status(kind, message)
    safe = html.escape(message)
    slot.markdown(
        f'<div class="app-status app-status--{kind}" role="status">{safe}</div>',
        unsafe_allow_html=True,
    )


def _render_document_status(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
    slot=None,
):
    """Show one status line so the visitor knows when they can ask."""
    kind, message = _derive_document_status(
        uploaded_file=uploaded_file,
        resolved_upload=resolved_upload,
    )
    target = slot if slot is not None else st.empty()
    paint_status(target, kind, message)
    if kind == "ready" and st.session_state.developer_mode:
        stats = st.session_state.get("indexed_doc_stats") or {}
        pages = stats.get("pages", 0)
        parts = []
        if pages:
            parts.append(f"{pages} page{'s' if pages != 1 else ''}")
        if stats.get("chunks") is not None:
            parts.append(f"{stats['chunks']} chunks")
        if stats.get("header_split") is not None:
            parts.append(f"header split {'ON' if stats['header_split'] else 'OFF'}")
        if parts:
            st.caption(" · ".join(parts))
        dev_logs = st.session_state.get("dev_index_logs") or {}
        if dev_logs.get("chunks"):
            st.caption(dev_logs["chunks"])
        if dev_logs.get("faiss"):
            st.caption(dev_logs["faiss"])
    return target


def _reset_chat_state() -> None:
    """Clear question history while keeping the indexed document."""
    st.session_state.last_query = None
    st.session_state.last_retrieved_docs = []
    st.session_state.last_retrieval_mode = None
    st.session_state.last_raw_results = []
    st.session_state.last_context = ""
    st.session_state.last_answer = None
    st.session_state.last_retrieval_metrics = None
    st.session_state.last_generation_error = None
    st.session_state.last_response_timing = None
    st.session_state.messages = []


def _reset_document_state(*, bump_uploader_key: bool = False) -> None:
    """Clear the index, the cached upload, and the chat."""
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.current_file = None
    st.session_state.last_processed_name = None
    st.session_state.last_processed_hash = None
    st.session_state.bm25_state = None
    st.session_state.indexed_doc_stats = None
    st.session_state.last_generation_error = None
    st.session_state.sample_preload_failed = False
    _clear_dev_index_logs()
    _clear_cached_upload()
    _reset_chat_state()
    _set_doc_status("empty", _CLIENT_COPY["doc_empty"])
    if bump_uploader_key:
        st.session_state.uploader_key_version += 1


def _set_processed_document(file_name: str, file_hash: str) -> None:
    """Store which file is indexed and start a fresh chat."""
    st.session_state.last_processed_name = file_name
    st.session_state.last_processed_hash = file_hash
    st.session_state.current_file = file_name
    _reset_chat_state()


def _init_session_state(
    *,
    developer_mode: bool,
    dummy_generator_only: bool,
    retrieval_strategy_default: str,
    reranker_enabled_default: bool,
) -> None:
    """Create session keys once so later wiring can read them."""
    defaults = {
        "vector_store": None,
        "chunks": None,
        "current_file": None,
        "last_processed_name": None,
        "last_processed_hash": None,
        "uploader_key_version": 0,
        "last_query": None,
        "last_retrieved_docs": [],
        "last_retrieval_mode": None,
        "last_raw_results": [],
        "last_context": "",
        "last_answer": None,
        "last_retrieval_metrics": None,
        "messages": [],
        "developer_mode": developer_mode,
        "enable_ocr": True,
        "use_page_separators": True,
        "dummy_generator_only": dummy_generator_only,
        "bm25_state": None,
        "retrieval_strategy": (
            retrieval_strategy_default
            if retrieval_strategy_default in {"semantic", "hybrid"}
            else "semantic"
        ),
        "enable_reranker": reranker_enabled_default if developer_mode else False,
        "indexed_doc_stats": None,
        "dev_index_logs": None,
        "last_generation_error": None,
        "last_response_timing": None,
        "_uploader_widget_had_file": False,
        "doc_status": "empty",
        "doc_status_detail": _CLIENT_COPY["doc_empty"],
        "ask_box": "",
        "own_pdf_mode": False,
        "show_own_pdf": False,
        "open_page_preview": None,
        "sample_preload_failed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
