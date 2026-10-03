"""Session keys, upload cache, and document-ready helpers for the page script."""

from __future__ import annotations

import hashlib
import os

import streamlit as st

from page_copy import _CLIENT_COPY

# Stable for the lifetime of the Streamlit process (survives reruns, changes on container restart).
_APP_BOOT_ID = str(os.getpid())


def _on_new_browser_session() -> None:
    """Drop a stale upload when the browser session is new, so a ghost name is not ready."""
    if st.session_state.get("_app_boot_id") == _APP_BOOT_ID:
        return
    st.session_state._app_boot_id = _APP_BOOT_ID
    st.session_state.uploader_key_version = (
        int(st.session_state.get("uploader_key_version", 0)) + 1
    )
    st.session_state._fresh_session_hint = True
    for key in ("uploaded_pdf_bytes", "uploaded_pdf_name", "uploaded_pdf_hash"):
        st.session_state.pop(key, None)
    st.session_state.last_processed_name = None
    st.session_state.last_processed_hash = None
    st.session_state.current_file = None
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.indexed_doc_stats = None
    st.session_state.bm25_state = None


def _upload_in_flight(*, uploaded_file, resolved_upload: tuple[bytes, str] | None) -> bool:
    """True when we have file bytes but no searchable index yet."""
    if _document_is_indexed():
        return False
    return resolved_upload is not None or uploaded_file is not None or bool(
        st.session_state.get("uploaded_pdf_bytes")
    )


def _set_indexed_doc_stats(
    *,
    file_name: str,
    pages: int,
    chars: int,
    reused_index: bool,
    header_split: bool,
    chunks: int | None = None,
) -> None:
    """Remember page counts for the ready line."""
    st.session_state.indexed_doc_stats = {
        "file_name": file_name,
        "pages": pages,
        "chars": chars,
        "reused_index": reused_index,
        "chunks": chunks,
        "header_split": header_split,
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
    if st.session_state.current_file or st.session_state.get("uploaded_pdf_name"):
        return _CLIENT_COPY["chat_blocked_ghost"]
    return _CLIENT_COPY["chat_blocked_empty"]


def _render_document_status(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
) -> None:
    """Show indexing or ready so the visitor knows when they can ask."""
    stats = st.session_state.get("indexed_doc_stats")
    if stats and _document_is_indexed():
        name = stats.get("file_name") or st.session_state.current_file or "Document"
        pages = stats.get("pages", 0)
        st.success(_CLIENT_COPY["doc_ready"].format(name=name))
        if st.session_state.developer_mode:
            dev_logs = st.session_state.get("dev_index_logs") or {}
            if dev_logs.get("chunks"):
                st.success(dev_logs["chunks"])
            if dev_logs.get("faiss"):
                st.success(dev_logs["faiss"])
            detail = f"{pages} page{'s' if pages != 1 else ''} indexed"
            if stats.get("chars"):
                detail += f" · {stats['chars']:,} characters extracted"
            if stats.get("reused_index"):
                detail += " · index reused (no re-processing)"
            if stats.get("chunks") is not None:
                detail += f" · {stats['chunks']} chunks"
            if stats.get("header_split") is not None:
                detail += f" · header split {'ON' if stats['header_split'] else 'OFF'}"
            st.caption(detail)
        elif pages:
            st.caption(f"{pages} page{'s' if pages != 1 else ''} processed")
        return

    if _document_is_indexed() and st.session_state.current_file:
        st.success(
            _CLIENT_COPY["doc_ready"].format(name=st.session_state.current_file)
        )
        return

    if st.session_state.get("uploaded_pdf_name") and not _document_is_indexed():
        st.info(
            _CLIENT_COPY["doc_indexing"].format(
                name=st.session_state.uploaded_pdf_name
            )
        )
        return

    if _upload_in_flight(
        uploaded_file=uploaded_file, resolved_upload=resolved_upload
    ):
        st.warning(_CLIENT_COPY["doc_stale"])


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
    _clear_dev_index_logs()
    _clear_cached_upload()
    _reset_chat_state()
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
        # Local / VPS: Ollama by default. Set USE_DUMMY_GENERATOR=true for UI placeholder tests.
        "dummy_generator_only": dummy_generator_only,
        "bm25_state": None,
        "retrieval_strategy": (
            retrieval_strategy_default
            if retrieval_strategy_default in {"semantic", "hybrid"}
            else "semantic"
        ),
        "enable_reranker": reranker_enabled_default,
        "indexed_doc_stats": None,
        "dev_index_logs": None,
        "last_generation_error": None,
        "last_response_timing": None,
        "_uploader_widget_had_file": False,
        "_scroll_chat_to_bottom": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
