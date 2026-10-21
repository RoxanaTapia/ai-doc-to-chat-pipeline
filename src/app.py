import os

# Before tokenizers/sentence-transformers import: avoid a first-ask deadlock.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import html
import logging
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from api_client import api_base_url, chat_via_api
from ask_runtime import AskTimeoutError, log_ask_stage, remaining_timeout, run_with_timeout
from page_copy import (
    _CLIENT_COPY,
    STARTER_CHIPS,
    _render_booking_cta,
    _render_client_footer,
    _render_client_hero,
    _render_privacy_badge,
    _render_sidebar_footer,
    _render_sidebar_pitch,
    booking_url,
    should_show_starter_chips,
)
from rag import (
    generate_answer_stream,
    load_generation_config,
    resolve_llm_provider_name,
)
from rag.chunking import apply_hard_section_context_filter, chunk_pages, extract_target_section
from rag.citations import (
    INSUFFICIENT_CONTEXT_ANSWER,
    assemble_context,
    context_sufficient_for_query,
    is_insufficient_answer,
    visitor_refusal_answer,
)
from rag.ingestion import extract_pdf
from rag.retrieval import (
    RetrievalConfig,
    build_bm25_index,
    finalize_retrieval,
    first_stage_retrieval,
)
from session_upload import (
    SAMPLE_NDA_NAME,
    _cache_upload,
    _chat_blocked_user_message,
    _chat_input_placeholder,
    _document_is_indexed,
    _init_session_state,
    _on_new_browser_session,
    _reading_message,
    _ready_message,
    _render_document_status,
    _reset_document_state,
    _resolve_upload,
    _set_dev_index_logs,
    _set_indexed_doc_stats,
    _set_processed_document,
    mark_sample_preload_failed,
    mark_upload_unreadable,
    page_text_counts,
    paint_status,
    preload_sample_nda,
    render_sample_preload_error,
)
from sources_panel import (
    _build_sources_payload,
    _dev_panel_title,
    _paired_turns,
    _question_preview,
    _remember_response_timing,
    _render_answer_card,
    _render_eval_context_panel,
    _render_source_checklist,
)

st.set_page_config(
    page_title="Ask your document",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)
APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=APP_ROOT / ".env")
MAX_CHAT_MESSAGES = 40
logger = logging.getLogger(__name__)


def _presentation_mode() -> str:
    """
    Return 'client' or 'developer'.
    Override with APP_PRESENTATION_MODE=client|developer.
    Default is client (pilot demos); use developer for local tuning.
    """
    override = (os.getenv("APP_PRESENTATION_MODE") or "").strip().lower()
    if override in {"client", "developer"}:
        return override
    return "client"


def _dev_toggle_allowed() -> bool:
    """Whether the sidebar shows the developer-mode toggle (opt-in only)."""
    return _env_bool("APP_ALLOW_DEV_TOGGLE", False)


def _human_model_label(provider: str, model: str) -> str:
    """Buyer-facing model name; keep API ids out of the sidebar caption."""
    raw = (model or "").strip()
    if provider == "anthropic":
        lower = raw.lower()
        if "haiku-4-5" in lower or lower.startswith("claude-haiku-4-5"):
            return "Claude Haiku 4.5"
        if "haiku" in lower:
            return "Claude Haiku"
        if "sonnet" in lower:
            return "Claude Sonnet"
        if "opus" in lower:
            return "Claude Opus"
        return "Claude"
    if provider == "ollama":
        return raw or "local model"
    return raw or provider


def _active_generator_label(dummy_mode: bool) -> str:
    """Calm caption for the active generation backend (no secrets)."""
    provider = resolve_llm_provider_name(dummy_mode=dummy_mode)
    if provider == "dummy":
        return "Dummy · UI placeholder (no LLM)"
    settings = load_generation_config()
    if provider == "anthropic":
        model_id = str(settings.get("anthropic_model") or "")
        return f"Anthropic · {_human_model_label(provider, model_id)}"
    if provider == "ollama":
        model_id = str(settings.get("model") or "")
        return f"Ollama · {_human_model_label(provider, model_id)}"
    return provider


def _apply_presentation_mode_lock() -> None:
    """
    Pin UI mode from APP_PRESENTATION_MODE when the dev toggle is hidden.
    Prevents clients on VPS from seeing or enabling developer controls.
    """
    if _dev_toggle_allowed():
        return
    st.session_state.developer_mode = _presentation_mode() == "developer"


def _apply_client_ask_guards() -> None:
    """Skip the cross-encoder on the pilot so the first question can reach the model."""
    if st.session_state.developer_mode:
        return
    st.session_state.enable_reranker = False


# Load configuration
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"

try:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing config file at {CONFIG_PATH}. "
            "Create it with chunking and embeddings settings."
        )

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    CHUNK_SIZE = config["chunking"]["chunk_size"]
    CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]
    SPLIT_ON_LEGAL_HEADERS = bool(config["chunking"].get("split_on_legal_headers", True))
    EMBEDDING_MODEL = config["embeddings"]["model_name"]
    rag_cfg = config.get("rag", {}) or {}
    retrieval_cfg = rag_cfg.get("retrieval", {}) or {}
    reranker_cfg = retrieval_cfg.get("reranker", {}) or {}

    TOP_K = int(rag_cfg.get("top_k", 4))
    FETCH_K = int(rag_cfg.get("fetch_k", 50))
    EARLY_PAGE_MAX = int(rag_cfg.get("early_page_max", 20))

    RETRIEVAL_STRATEGY_DEFAULT = str(retrieval_cfg.get("strategy", "semantic")).strip().lower()
    BM25_FETCH_K = int(retrieval_cfg.get("bm25_fetch_k", FETCH_K))
    RRF_K = int(retrieval_cfg.get("rrf_k", 60))
    DENSE_WEIGHT = float(retrieval_cfg.get("dense_weight", 1.0))
    BM25_WEIGHT = float(retrieval_cfg.get("bm25_weight", 1.0))

    RERANKER_ENABLED_DEFAULT = bool(reranker_cfg.get("enabled", False))
    RERANKER_MODEL_NAME = str(reranker_cfg.get("model_name", "BAAI/bge-reranker-base"))
    RERANKER_TOP_N = int(reranker_cfg.get("top_n", 50))

    section_cfg = retrieval_cfg.get("section_aware", {}) or {}
    SECTION_AWARE_ENABLED = bool(section_cfg.get("enabled", True))
    SECTION_AWARE_BOOST = float(section_cfg.get("boost", 1.5))
    SECTION_AWARE_MIN_CHUNKS = int(section_cfg.get("min_chunks", 2))
    SECTION_HARD_CONTEXT_FILTER = bool(section_cfg.get("hard_context_filter", True))
    SECTION_CONTEXT_MIN_CHUNKS = int(section_cfg.get("context_min_chunks", 2))
    MAX_CHUNKS_PER_PAGE = int(retrieval_cfg.get("max_chunks_per_page", 2))
    DEDUPE_SIMILAR_CHUNKS = bool(retrieval_cfg.get("dedupe_similar_chunks", True))
    DEDUPE_PREFIX_CHARS = int(retrieval_cfg.get("dedupe_prefix_chars", 180))
    CONTEXT_SUFFICIENCY_GUARD = bool(retrieval_cfg.get("context_sufficiency_guard", True))

    RETRIEVAL_CONFIG = RetrievalConfig(
        top_k=TOP_K,
        fetch_k=FETCH_K,
        early_page_max=EARLY_PAGE_MAX,
        bm25_fetch_k=BM25_FETCH_K,
        rrf_k=RRF_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        reranker_top_n=RERANKER_TOP_N,
        section_aware_enabled=SECTION_AWARE_ENABLED,
        section_aware_boost=SECTION_AWARE_BOOST,
        section_aware_min_chunks=SECTION_AWARE_MIN_CHUNKS,
        max_chunks_per_page=MAX_CHUNKS_PER_PAGE,
        dedupe_similar_chunks=DEDUPE_SIMILAR_CHUNKS,
        dedupe_prefix_chars=DEDUPE_PREFIX_CHARS,
    )

    ui_cfg = rag_cfg.get("ui", {}) or {}
    SOURCES_DISPLAY_MAX = int(ui_cfg.get("sources_display_max", 3))
    SOURCE_PREVIEW_CHARS = int(ui_cfg.get("source_preview_chars", 280))
except (FileNotFoundError, OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
    st.error(f"Configuration error in {CONFIG_PATH}: {exc}")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    """Cache embedding model to avoid repeated heavy initialization."""
    if _env_bool("USE_STUB_EMBEDDINGS", False):
        from langchain_core.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=32)
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_sample_nda_index(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    header_split: bool,
) -> dict:
    """Index the walkthrough NDA once per process so first paint can skip setup."""
    from rag.ingestion import extract_pdf as _extract_pdf
    from session_upload import sample_nda_path

    path = sample_nda_path()
    extracted = _extract_pdf(path, enable_ocr=False)
    chunks = chunk_pages(
        extracted.page_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_on_legal_headers=header_split,
    )
    store = FAISS.from_documents(chunks, get_embeddings(model_name))
    text_pages, blank_pages = page_text_counts(extracted.page_docs)
    return {
        "bytes": path.read_bytes(),
        "chunks": chunks,
        "vector_store": store,
        "pages": len(extracted.page_docs),
        "text_pages": text_pages,
        "blank_pages": blank_pages,
        "chars": len(extracted.preview_text.strip()),
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@st.cache_resource(show_spinner=False)
def _get_cross_encoder(model_name: str):
    """Cache cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _append_chat_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    timing: dict[str, float] | None = None,
    for_question: str | None = None,
    eval_context: str | None = None,
    target_section: str | None = None,
    eval_chunk_count: int | None = None,
) -> None:
    """Append a chat message and prune old history."""
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    if role == "assistant":
        message["refusal"] = bool(is_insufficient_answer(content, for_question))
    if role == "assistant" and timing and timing.get("total_ms", 0) > 0:
        message["timing"] = {
            "total_ms": timing["total_ms"],
            "retrieval_ms": timing.get("retrieval_ms", 0.0),
            "generation_ms": timing.get("generation_ms", 0.0),
        }
    if role == "assistant" and for_question:
        message["for_question"] = _question_preview(for_question)
    if role == "assistant" and st.session_state.developer_mode:
        if eval_context:
            message["eval_context"] = eval_context
        if target_section:
            message["target_section"] = target_section
        if eval_chunk_count is not None:
            message["eval_chunk_count"] = eval_chunk_count
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > MAX_CHAT_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_MESSAGES:]


def _ensure_bm25_index() -> tuple[object | None, str | None]:
    """Build and cache BM25 index for the current chunk set."""
    chunks = st.session_state.chunks or []
    if st.session_state.get("bm25_state") is not None:
        state = st.session_state.bm25_state
        if state.get("chunk_count") == len(chunks):
            return state.get("index"), None
    index, warning = build_bm25_index(chunks)
    if index is not None:
        st.session_state.bm25_state = {"index": index, "chunk_count": len(chunks)}
    return index, warning


def _rerank_predict(pairs):
    return _get_cross_encoder(RERANKER_MODEL_NAME).predict(pairs)


def _run_first_stage_retrieval(
    query: str,
    *,
    candidate_limit: int,
) -> tuple[list[tuple[Document, float]], str, str | None, dict[str, int]]:
    bm25_index, bm25_warning = _ensure_bm25_index()
    results, mode, warning, diag = first_stage_retrieval(
        query,
        strategy=st.session_state.retrieval_strategy,
        vector_store=st.session_state.vector_store,
        chunks=st.session_state.chunks or [],
        bm25_index=bm25_index,
        bm25_warning=bm25_warning,
        config=RETRIEVAL_CONFIG,
        candidate_limit=candidate_limit,
    )
    if diag.get("used_global_fallback") and st.session_state.developer_mode:
        st.caption(
            f"No matches found in pages <= {EARLY_PAGE_MAX}. Showing best matches from all pages."
        )
    return results, mode, warning, diag


def _finalize_retrieval_candidates(
    query: str,
    *,
    candidate_pool: list[tuple[Document, float]],
    mode: str,
) -> tuple[list[tuple[Document, float]], str, str | None, float]:
    rerank_elapsed_ms = 0.0
    rerank_predict = _rerank_predict if st.session_state.enable_reranker else None
    if rerank_predict is not None:
        rerank_start = time.perf_counter()
    pool, mode, warning = finalize_retrieval(
        query,
        candidate_pool=candidate_pool,
        mode=mode,
        enable_reranker=st.session_state.enable_reranker,
        rerank_predict=rerank_predict,
        config=RETRIEVAL_CONFIG,
    )
    if rerank_predict is not None:
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000.0
    return pool, mode, warning, rerank_elapsed_ms


def _ollama_recommended_models() -> list[str]:
    try:
        model_name = load_generation_config().get("model", "llama3.1:8b")
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
        model_name = "llama3.1:8b"
    preferred = [model_name, "llama3.1:8b", "phi3.5:latest", "phi3:mini"]
    return list(dict.fromkeys(preferred))


def _ollama_recovery_expander(*, expanded: bool = False) -> None:
    """Terminal commands and model hints; no duplicate banners."""
    recommended_models = _ollama_recommended_models()
    pull_commands = "\n".join(f"ollama pull {name}" for name in recommended_models)
    recommended_order_text = " → ".join(f"`{name}`" for name in recommended_models)
    with st.expander("How to fix local generation (Ollama)", expanded=expanded):
        st.markdown(
            "**Metal / GPU crashes on macOS** often come from the Ollama runner. "
            "Update Ollama, try another model/quantization, or reduce context "
            "(`OLLAMA_NUM_CTX` in `.env`). See `docs/ollama-troubleshooting.md`."
        )
        st.markdown("Run these commands in a terminal:")
        st.code(
            f"ollama serve\n{pull_commands}\nollama list",
            language="bash",
        )
        st.caption(f"CPU smoke-test order: {recommended_order_text}.")


def _active_provider_for_errors() -> str:
    """Provider name used when mapping generation failures to user-facing copy."""
    return resolve_llm_provider_name(dummy_mode=st.session_state.get("dummy_generator_only", True))


def _generation_failure_reason(exc: BaseException) -> tuple[str, str]:
    """Return (short_user_reason, normalized_error_text)."""
    error_text = str(exc).lower()
    provider = _active_provider_for_errors()

    if provider == "anthropic":
        if "api_key" in error_text or "authentication" in error_text or "401" in error_text:
            reason = "Anthropic rejected the API key. Check `ANTHROPIC_API_KEY` in `.env`."
        elif "temperature" in error_text and "top_p" in error_text:
            reason = (
                "Anthropic rejected the request (sampling settings). "
                "Restart the app after updating, then retry."
            )
        elif "rate" in error_text or "429" in error_text:
            reason = "Anthropic rate limit reached. Wait a moment, then retry."
        elif "timeout" in error_text or "timed out" in error_text:
            reason = "Anthropic timed out. Retry with a shorter question."
        else:
            detail = str(exc).strip()
            # Prefer the API message body when present; keep it short for chat.
            if "message': '" in detail:
                try:
                    reason = (
                        "Anthropic error: " + detail.split("message': '", 1)[1].split("'", 1)[0]
                    )
                except IndexError:
                    reason = "Could not generate an Anthropic answer right now."
            else:
                reason = "Could not generate an Anthropic answer right now."
        return reason, error_text

    try:
        model_name = load_generation_config().get("model", "llama3.1:8b")
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
        model_name = "llama3.1:8b"
    if "connection" in error_text or "refused" in error_text:
        reason = "Could not connect to Ollama. Start the Ollama server first."
    elif "not found" in error_text or "model" in error_text:
        reason = f"The model `{model_name}` is unavailable locally. Pull it, then retry."
    elif "timeout" in error_text or "timed out" in error_text:
        reason = "Local model timed out. Retry with a shorter question or smaller context."
    elif (
        "mtllibrary" in error_text
        or "metal" in error_text
        or "llama runner process has terminated" in error_text
    ):
        reason = (
            "Ollama’s local model runner crashed (often a Metal/GPU issue on macOS). "
            "Update Ollama, try a smaller model, or lower `OLLAMA_NUM_CTX`."
        )
    else:
        reason = "Could not generate an Ollama answer right now."
    return reason, error_text


def _render_persistent_generation_error() -> None:
    """Survives reruns (e.g. sidebar toggles) until dismissed or a new success."""
    err = st.session_state.get("last_generation_error")
    if not err:
        return
    reason = err.get("user_reason", "Generation failed.")
    st.error(reason)
    if _active_provider_for_errors() == "ollama":
        _ollama_recovery_expander(expanded=False)
    detail = err.get("detail")
    if st.session_state.developer_mode and detail:
        with st.expander("Technical details", expanded=False):
            st.code(detail, language="text")
    if st.button("Dismiss", key="dismiss_generation_error_banner"):
        st.session_state.last_generation_error = None
        st.rerun()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a fallback default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized == "":
        return default
    return normalized in {"1", "true", "yes", "on"}


def _ask_timeout_seconds() -> float:
    """Wall-clock cap for one look-up; matches generation.timeout_seconds."""
    try:
        return max(1.0, float(load_generation_config().get("timeout_seconds", 90)))
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
        return 90.0


def _queue_question(text: str) -> bool:
    """Remember a question to run on the next pass. True when it was non-empty."""
    question = (text or "").strip()
    if not question:
        return False
    st.session_state.queued_question = question
    st.session_state._clear_ask_box = True
    return True


def _on_chip_click(question: str) -> None:
    """Queue in the widget callback so busy UI lands on the same run as typed Enter."""
    _queue_question(question)


def _on_ask_box_commit() -> None:
    """Queue on Enter or blur. The first Ask click is often only this commit."""
    if not _document_is_indexed():
        return
    if _queue_question(st.session_state.get("ask_box") or ""):
        st.session_state.ask_box = ""


def _continue_queued_ask(already_taken: str | None) -> None:
    """If on_change queued after we popped, rerun so that pass can look it up."""
    if already_taken:
        return
    if st.session_state.get("queued_question"):
        st.rerun()


_ASK_SHELL_EMITTED_KEY = "_ask_shell_emitted"


def _render_starter_chips(
    *,
    chat_ready: bool,
    query: str | None,
    shell_ready: bool,
):
    """Paint starter chips in a replaceable slot; empty until chat is ready.

    Always occupy this slot with ``st.empty()`` so a chip click (or the
    look-up pass) can clear the previous buttons before slow work starts.
    Without the placeholder, Streamlit keeps last-run chips on screen,
    greyed, under "Looking through the document…".

    Chips are emitted only after a prior run has already painted the hero and
    question input (``shell_ready``). A CSS gate is not enough: theme CSS
    arrives via ``st.markdown`` after early deltas, so chip widgets must not
    exist in the first paint stream. The reserve placeholder keeps height
    stable on that shell-only pass.
    """
    slot = st.empty()
    if not should_show_starter_chips(st.session_state.messages, query, chat_ready=chat_ready):
        return slot
    if not shell_ready:
        with slot.container():
            st.markdown(
                '<div class="app-chips-reserve" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )
        return slot
    with slot.container():
        st.markdown(
            '<p class="app-chips-label app-chip-mark">Try a starter question</p>',
            unsafe_allow_html=True,
        )
        for chip in STARTER_CHIPS:
            st.button(
                chip["label"],
                key=f"chip_{chip['label']}",
                on_click=_on_chip_click,
                args=(chip["question"],),
            )
    return slot


def _collect_generated_answer(context: str, query: str, dummy_mode: bool) -> str:
    """Join token chunks in-process. A live UI stream can hang on the next Ask."""
    if api_base_url():
        return chat_via_api(context=context, query=query, dummy_mode=dummy_mode)
    return "".join(generate_answer_stream(context=context, query=query, dummy_mode=dummy_mode))


def _render_ocr_status(
    *,
    enable_ocr: bool,
    scanned_pages_detected: int,
    ocr_pages_attempted: int,
    ocr_pages_used: int,
    ocr_warning: str | None,
    developer_mode: bool,
) -> None:
    """Render OCR diagnostics and user-facing warnings after extraction."""
    if not enable_ocr:
        return

    if scanned_pages_detected > 0 and developer_mode:
        ocr_missed_pages = max(0, ocr_pages_attempted - ocr_pages_used)
        st.caption(
            "OCR diagnostics: "
            f"scanned pages detected={scanned_pages_detected}, "
            f"OCR applied={ocr_pages_used}, "
            f"OCR unresolved={ocr_missed_pages}"
        )

    if not developer_mode:
        return

    if ocr_pages_used > 0:
        st.caption(f"OCR used on {ocr_pages_used} page(s). Results may vary by scan quality.")
    elif scanned_pages_detected > 0 and ocr_warning:
        st.caption(ocr_warning)


_init_session_state(
    developer_mode=_presentation_mode() == "developer",
    dummy_generator_only=_env_bool("USE_DUMMY_GENERATOR", False),
    retrieval_strategy_default=RETRIEVAL_STRATEGY_DEFAULT,
    reranker_enabled_default=RERANKER_ENABLED_DEFAULT,
)
_on_new_browser_session()
_apply_presentation_mode_lock()
_apply_client_ask_guards()

_github = "https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline"
_show_sidebar = bool(st.session_state.developer_mode or _dev_toggle_allowed())

if _show_sidebar:
    _render_sidebar_pitch()
    st.sidebar.link_button("Exit demo", "/invite/exit")
    if _dev_toggle_allowed() or st.session_state.developer_mode:
        st.sidebar.divider()
    if _dev_toggle_allowed():
        st.session_state.developer_mode = st.sidebar.toggle(
            "Developer mode (show retrieval debug)",
            value=st.session_state.developer_mode,
            help=(
                "ON: show retrieval ranking and chunk diagnostics. "
                "OFF: cleaner client-facing presentation."
            ),
        )
        st.sidebar.caption(
            f"Presentation mode: {'Developer' if st.session_state.developer_mode else 'Client'}"
        )
    if st.session_state.developer_mode:
        with st.sidebar.expander("Advanced options", expanded=False):
            st.session_state.enable_ocr = st.toggle(
                "Enable OCR for scanned pages",
                value=st.session_state.enable_ocr,
                help="Attempts OCR only on pages with little/no extractable text.",
            )
            st.session_state.use_page_separators = st.checkbox(
                "Use page separators in context",
                value=st.session_state.use_page_separators,
                help="Adds page labels between chunks when assembling context for generation.",
            )
            st.session_state.dummy_generator_only = st.checkbox(
                "Use dummy generator only (for testing)",
                value=st.session_state.dummy_generator_only,
                help=(
                    "Used only when LLM_PROVIDER is unset. ON: placeholder answers. "
                    "OFF: Ollama. Non-empty LLM_PROVIDER (e.g. anthropic) always wins."
                ),
            )
            st.caption(
                f"**Chunking:** header split **{'ON' if SPLIT_ON_LEGAL_HEADERS else 'OFF'}** "
                f"· size={CHUNK_SIZE} · overlap={CHUNK_OVERLAP}"
            )
            st.caption(
                f"**Retrieval defaults:** {RETRIEVAL_STRATEGY_DEFAULT} · "
                f"dedupe {'ON' if DEDUPE_SIMILAR_CHUNKS else 'OFF'} · "
                f"context guard {'ON' if CONTEXT_SUFFICIENCY_GUARD else 'OFF'}"
            )
        st.session_state.retrieval_strategy = st.sidebar.selectbox(
            "Retrieval strategy",
            options=["semantic", "hybrid"],
            index=0 if st.session_state.retrieval_strategy == "semantic" else 1,
            help=(
                "semantic: dense vector search only. "
                "hybrid: dense + BM25 with reciprocal rank fusion."
            ),
        )
        st.session_state.enable_reranker = st.sidebar.checkbox(
            "Enable cross-encoder reranker",
            value=st.session_state.enable_reranker,
            help=(
                "Second-stage reranking of top candidates using a cross-encoder "
                f"({RERANKER_MODEL_NAME})."
            ),
        )
        _dev_provider = resolve_llm_provider_name(st.session_state.dummy_generator_only)
        _dev_settings = load_generation_config()
        _dev_model = (
            _dev_settings.get("anthropic_model")
            if _dev_provider == "anthropic"
            else _dev_settings.get("model")
        )
        if _dev_model:
            st.sidebar.caption(f"API id: `{_dev_model}`")
    _render_sidebar_footer(
        generator_label=_active_generator_label(st.session_state.dummy_generator_only),
        github_url=_github,
    )

if preload_sample_nda():
    try:
        cached = get_sample_nda_index(
            EMBEDDING_MODEL,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            SPLIT_ON_LEGAL_HEADERS,
        )
        _cache_upload(cached["bytes"], SAMPLE_NDA_NAME)
        st.session_state.chunks = cached["chunks"]
        st.session_state.vector_store = cached["vector_store"]
        st.session_state.bm25_state = None
        _set_processed_document(SAMPLE_NDA_NAME, cached["hash"])
        _set_indexed_doc_stats(
            file_name=SAMPLE_NDA_NAME,
            pages=cached["pages"],
            chars=cached["chars"],
            reused_index=True,
            header_split=SPLIT_ON_LEGAL_HEADERS,
            chunks=len(cached["chunks"]),
            text_pages=cached.get("text_pages"),
            blank_pages=cached.get("blank_pages"),
        )
    except (OSError, ValueError, RuntimeError, ImportError):
        mark_sample_preload_failed()

_render_client_hero(hide_sidebar=not _show_sidebar)

st.markdown('<div class="app-toolbar-mark" aria-hidden="true"></div>', unsafe_allow_html=True)
privacy_col, status_col, own_col = st.columns([1.1, 2.2, 1.4])
with privacy_col:
    _render_privacy_badge(
        resolve_llm_provider_name(dummy_mode=st.session_state.dummy_generator_only)
    )
status_slot = status_col.empty()
with own_col:
    st.markdown('<div class="app-own-pdf">', unsafe_allow_html=True)
    if st.button(
        _CLIENT_COPY["own_pdf_link"],
        key="own_pdf_toggle",
        help=_CLIENT_COPY["own_pdf_help"],
    ):
        st.session_state.show_own_pdf = not bool(st.session_state.get("show_own_pdf"))
        st.session_state.own_pdf_mode = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

uploader_key = f"pdf_uploader_{st.session_state.uploader_key_version}"
uploaded_file = None
if st.session_state.get("show_own_pdf") or st.session_state.get("own_pdf_mode"):
    uploaded_file = st.file_uploader(
        "PDF",
        type=["pdf"],
        key=uploader_key,
        label_visibility="collapsed",
        help=_CLIENT_COPY["own_pdf_help"],
    )

prev_uploader_had_file = st.session_state.get("_uploader_widget_had_file", False)
curr_uploader_has_file = uploaded_file is not None
if prev_uploader_had_file and not curr_uploader_has_file:
    if _document_is_indexed() or st.session_state.get("uploaded_pdf_bytes"):
        _reset_document_state(bump_uploader_key=True)
        st.session_state._uploader_widget_had_file = False
        st.session_state.own_pdf_mode = False
        st.session_state.show_own_pdf = False
        st.rerun()
st.session_state._uploader_widget_had_file = curr_uploader_has_file

extracted_text = ""
resolved_upload = _resolve_upload(uploaded_file)
_render_document_status(
    uploaded_file=uploaded_file,
    resolved_upload=resolved_upload,
    slot=status_slot,
)
render_sample_preload_error()

if resolved_upload is not None:
    file_bytes, file_name = resolved_upload
    uploaded_hash = _cache_upload(file_bytes, file_name)

    if uploaded_hash == st.session_state.last_processed_hash and _document_is_indexed():
        st.session_state.current_file = file_name
        prev = st.session_state.indexed_doc_stats or {}
        _set_indexed_doc_stats(
            file_name=file_name,
            pages=int(prev.get("pages", 0)),
            chars=int(prev.get("chars", 0)),
            reused_index=True,
            header_split=SPLIT_ON_LEGAL_HEADERS,
            chunks=prev.get("chunks"),
            text_pages=prev.get("text_pages"),
            blank_pages=prev.get("blank_pages"),
        )
    else:
        tmp_path = None
        try:
            paint_status(
                status_slot,
                "reading",
                _reading_message(file_name, _CLIENT_COPY["progress_prepare"]),
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = Path(tmp_file.name)

            paint_status(
                status_slot,
                "reading",
                _reading_message(file_name, _CLIENT_COPY["progress_extract"]),
            )

            extracted = extract_pdf(tmp_path, enable_ocr=st.session_state.enable_ocr)
            page_docs = extracted.page_docs
            extracted_text = extracted.preview_text
            paint_status(
                status_slot,
                "reading",
                _reading_message(file_name, _CLIENT_COPY["progress_chunk"]),
            )

            extracted_char_count = len(extracted_text.strip())
            text_pages, blank_pages = page_text_counts(page_docs)
            if st.session_state.developer_mode:
                st.caption(
                    f"Extraction: **{len(page_docs)}** pages · "
                    f"**{extracted_char_count:,}** characters"
                )
            _render_ocr_status(
                enable_ocr=st.session_state.enable_ocr,
                scanned_pages_detected=extracted.scanned_pages_detected,
                ocr_pages_attempted=extracted.ocr_pages_attempted,
                ocr_pages_used=extracted.ocr_pages_used,
                ocr_warning=extracted.ocr_warning,
                developer_mode=st.session_state.developer_mode,
            )
            if st.session_state.developer_mode and extracted_char_count > 0:
                with st.expander("Raw extraction preview (first 2000 chars)", expanded=False):
                    st.text_area(
                        "Extracted text sample",
                        extracted_text[:2000],
                        height=260,
                    )

            chunks = chunk_pages(
                page_docs,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                split_on_legal_headers=SPLIT_ON_LEGAL_HEADERS,
            )
            paint_status(
                status_slot,
                "reading",
                _reading_message(file_name, _CLIENT_COPY["progress_embed"]),
            )

            if st.session_state.developer_mode:
                split_note = " · header split ON" if SPLIT_ON_LEGAL_HEADERS else ""
                _set_dev_index_logs(
                    chunks_msg=(
                        f"Created {len(chunks)} chunks "
                        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}{split_note})"
                    ),
                )
            if not chunks:
                _reset_document_state()
                st.session_state.chunks = []
                _set_processed_document(file_name, uploaded_hash)
                _set_indexed_doc_stats(
                    file_name=file_name,
                    pages=len(page_docs),
                    chars=extracted_char_count,
                    reused_index=False,
                    header_split=SPLIT_ON_LEGAL_HEADERS,
                    text_pages=text_pages,
                    blank_pages=blank_pages,
                )
                paint_status(
                    status_slot,
                    "error",
                    "No extractable text in this PDF. Try a file with selectable text.",
                )
            else:
                if st.session_state.developer_mode:
                    with st.expander("First 3 chunks (debug view)", expanded=False):
                        for i, chunk in enumerate(chunks[:3], 1):
                            preview = (
                                chunk.page_content[:300] + "..."
                                if len(chunk.page_content) > 300
                                else chunk.page_content
                            )
                            st.markdown(
                                f"**Chunk {i}** ({len(chunk.page_content)} chars, "
                                f"page ~{chunk.metadata.get('page', 'N/A')}, "
                                f"start index: {chunk.metadata.get('start_index', 'N/A')})"
                            )
                            st.text(preview)

                st.session_state.chunks = chunks
                st.session_state.bm25_state = None

                had_existing_index = st.session_state.vector_store is not None
                paint_status(
                    status_slot,
                    "reading",
                    _reading_message(file_name, _CLIENT_COPY["progress_index"]),
                )
                start = time.time()
                embeddings = get_embeddings(EMBEDDING_MODEL)
                vector_store = FAISS.from_documents(
                    documents=st.session_state.chunks, embedding=embeddings
                )
                st.session_state.vector_store = vector_store
                took = time.time() - start

                if st.session_state.developer_mode:
                    _set_dev_index_logs(
                        faiss_msg=(
                            f"FAISS index {'re-' if had_existing_index else ''}created "
                            f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
                        ),
                    )

                _set_processed_document(file_name, uploaded_hash)
                _set_indexed_doc_stats(
                    file_name=file_name,
                    pages=len(page_docs),
                    chars=extracted_char_count,
                    reused_index=False,
                    header_split=SPLIT_ON_LEGAL_HEADERS,
                    chunks=len(chunks),
                    text_pages=text_pages,
                    blank_pages=blank_pages,
                )
                paint_status(status_slot, "ready", _ready_message(file_name))

        except (OSError, ValueError, RuntimeError):
            logger.exception("Could not read uploaded PDF %s", file_name)
            mark_upload_unreadable()
            paint_status(status_slot, "error", _CLIENT_COPY["pdf_unreadable"])
            # Clear the bad widget file and show Use sample NDA on the next paint.
            st.rerun()
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def _seed_capture_storyboard() -> None:
    """Paint the walkthrough answers for stills. Off unless DEMO_CAPTURE_STORYBOARD is set."""
    if not _env_bool("DEMO_CAPTURE_STORYBOARD", False):
        return
    if st.session_state.messages or not _document_is_indexed():
        return
    term = (
        "This Agreement shall remain in effect for a period of three (3) years "
        "from the date of signing."
    )
    law = (
        "This Agreement shall be governed by the laws of Spain. "
        "Any dispute arising hereunder shall be resolved exclusively in the courts of Madrid."
    )
    turns = (
        (
            STARTER_CHIPS[0]["question"],
            f"3 years\n\n{term}",
            [{"page": 2, "preview": f"Section 4 - Term {term}", "score": "N/A"}],
        ),
        (
            STARTER_CHIPS[1]["question"],
            f"Spain / Madrid\n\n{law}",
            [{"page": 3, "preview": f"Section 7 - Governing Law {law}", "score": "N/A"}],
        ),
        (
            STARTER_CHIPS[2]["question"],
            INSUFFICIENT_CONTEXT_ANSWER,
            [],
        ),
    )
    for question, answer, sources in turns:
        _append_chat_message("user", question)
        _append_chat_message("assistant", answer, sources=sources or None, for_question=question)


chat_ready = _document_is_indexed()
_seed_capture_storyboard()
if st.session_state.pop("_clear_ask_box", False):
    st.session_state.ask_box = ""
queued_question = st.session_state.pop("queued_question", None)

st.markdown('<p class="app-ask-mark">Ask</p>', unsafe_allow_html=True)
# No st.form: Streamlit 1.54 flashes "Missing Submit Button" while the form
# hydrates on first load (form delta arrives before the lazy submit button).
typed = st.text_input(
    "Question",
    key="ask_box",
    placeholder=_chat_input_placeholder(
        uploaded_file=uploaded_file,
        resolved_upload=resolved_upload,
    ),
    label_visibility="collapsed",
    disabled=not chat_ready,
    on_change=_on_ask_box_commit if chat_ready else None,
)
submitted = st.button(
    _CLIENT_COPY["ask_button"],
    key="ask_submit",
    type="primary",
    disabled=not chat_ready,
)
if submitted and _queue_question(typed):
    st.rerun()
# First Ask click is often only a blur/commit. on_change queues after the pop
# above, and Streamlit drops the button press. Continue that queued look-up.
_continue_queued_ask(queued_question)
query = (queued_question or "").strip() or None

# Ask shell (hero already above + this input) must exist for one full run
# before chip widgets are created, so they cannot paint ahead of the header.
shell_was_ready = bool(st.session_state.get(_ASK_SHELL_EMITTED_KEY))
st.session_state[_ASK_SHELL_EMITTED_KEY] = True

# Same slot every run. Empty until the document is indexed; emptied again
# before look-up so last-run chips cannot linger as stale widgets.
chip_slot = _render_starter_chips(
    chat_ready=chat_ready,
    query=query,
    shell_ready=shell_was_ready,
)
if (
    chat_ready
    and not shell_was_ready
    and should_show_starter_chips(st.session_state.messages, query, chat_ready=True)
):
    st.rerun()

if query and query.strip() and chat_ready:
    query = query.strip()
    chip_slot.empty()
    _append_chat_message("user", query)
    paint_status(status_slot, "answering", _CLIENT_COPY["doc_answering"])

    retrieval_error = None
    generation_error = None
    raw_results: list[tuple[Document, float]] = []
    retrieved_docs: list[Document] = []
    context = ""
    retrieval_warning: str | None = None
    retrieval_diag: dict[str, int] = {}
    retrieval_elapsed_ms = 0.0
    rerank_elapsed_ms = 0.0
    generation_elapsed_ms = 0.0
    total_elapsed_ms = 0.0

    with st.container():
        st.markdown(
            f'<p class="app-answer-q">{html.escape(query)}</p>',
            unsafe_allow_html=True,
        )
        timer_placeholder = st.empty()
        stop_timer = threading.Event()
        script_ctx = get_script_run_ctx()
        total_start = time.perf_counter()

        def _thinking_timer_loop() -> None:
            if script_ctx is not None:
                add_script_run_ctx(threading.current_thread(), script_ctx)
            t0 = time.perf_counter()
            while not stop_timer.wait(0.12):
                elapsed = time.perf_counter() - t0
                timer_placeholder.caption(f"Working… {elapsed:.1f}s")

        typing_slot = st.empty()
        typing_slot.markdown(
            f'<p class="app-typing">{html.escape(_CLIENT_COPY["doc_answering"])}'
            '<span class="app-typing-dots"></span></p>',
            unsafe_allow_html=True,
        )
        timer_thread = None
        if script_ctx is not None and st.session_state.developer_mode:
            timer_thread = threading.Thread(
                target=_thinking_timer_loop,
                daemon=True,
                name="thinking-timer",
            )
            timer_thread.start()
        try:
            retrieval_start = time.perf_counter()
            try:
                candidate_limit = (
                    max(TOP_K, RERANKER_TOP_N) if st.session_state.enable_reranker else TOP_K
                )
                bm25_index, bm25_warning = _ensure_bm25_index()
                strategy = st.session_state.retrieval_strategy
                vector_store = st.session_state.vector_store
                chunks = st.session_state.chunks or []
                dummy_mode = st.session_state.dummy_generator_only
                use_page_separators = bool(st.session_state.use_page_separators)
                timeout_seconds = _ask_timeout_seconds()
                log_ask_stage("start", question=query)

                def _retrieve_only():
                    return first_stage_retrieval(
                        query,
                        strategy=strategy,
                        vector_store=vector_store,
                        chunks=chunks,
                        bm25_index=bm25_index,
                        bm25_warning=bm25_warning,
                        config=RETRIEVAL_CONFIG,
                        candidate_limit=candidate_limit,
                    )

                candidate_pool, mode, first_stage_warning, retrieval_diag = run_with_timeout(
                    _retrieve_only,
                    remaining_timeout(total_start, timeout_seconds),
                )
                raw_results, final_mode, reranker_warning, rerank_elapsed_ms = (
                    _finalize_retrieval_candidates(
                        query,
                        candidate_pool=candidate_pool,
                        mode=mode,
                    )
                )
                st.session_state.last_retrieval_mode = final_mode
                retrieval_warning = reranker_warning or first_stage_warning
                retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0
                log_ask_stage(
                    "retrieval_done",
                    question=query,
                    elapsed_ms=retrieval_elapsed_ms,
                    chunks=len(raw_results),
                    mode=final_mode,
                )

                for doc, score in raw_results:
                    doc.metadata["similarity"] = score
                st.session_state.last_raw_results = raw_results

                context_results, hard_filter_warning = apply_hard_section_context_filter(
                    query,
                    raw_results,
                    all_chunks=st.session_state.chunks,
                    top_k=TOP_K,
                    enabled=SECTION_HARD_CONTEXT_FILTER,
                    min_chunks=SECTION_CONTEXT_MIN_CHUNKS,
                )
                if hard_filter_warning:
                    retrieval_warning = " · ".join(
                        part for part in (retrieval_warning, hard_filter_warning) if part
                    )
                    st.session_state.last_retrieval_mode = (
                        f"{st.session_state.last_retrieval_mode}_section_filtered"
                    )

                retrieved_docs = [doc for doc, _score in context_results]
                st.session_state.last_query = query
                st.session_state.last_retrieved_docs = retrieved_docs
                context = assemble_context(
                    context_results,
                    use_page_separators=use_page_separators,
                )
                st.session_state.last_context = context

                context_ok, _context_gap = context_sufficient_for_query(query, context)
                if CONTEXT_SUFFICIENCY_GUARD and not context_ok:
                    st.session_state.last_answer = INSUFFICIENT_CONTEXT_ANSWER
                    st.session_state.last_generation_error = None
                    st.session_state.last_retrieval_mode = (
                        f"{st.session_state.last_retrieval_mode}_context_guard"
                    )
                    generation_elapsed_ms = 0.0
                else:
                    generation_start = time.perf_counter()
                    log_ask_stage(
                        "model_call_start",
                        question=query,
                        elapsed_ms=retrieval_elapsed_ms,
                    )
                    try:
                        streamed = run_with_timeout(
                            lambda: _collect_generated_answer(context, query, dummy_mode),
                            remaining_timeout(total_start, timeout_seconds),
                        )
                        st.session_state.last_answer = (
                            streamed if isinstance(streamed, str) else None
                        )
                        st.session_state.last_generation_error = None
                    except AskTimeoutError as e:
                        generation_error = e
                        st.session_state.last_answer = None
                    except Exception as e:
                        generation_error = e
                        st.session_state.last_answer = None
                    generation_elapsed_ms = (time.perf_counter() - generation_start) * 1000.0
                    if generation_error is None:
                        log_ask_stage(
                            "model_call_end",
                            question=query,
                            elapsed_ms=generation_elapsed_ms,
                        )
                    else:
                        log_ask_stage(
                            "error",
                            question=query,
                            elapsed_ms=generation_elapsed_ms,
                            reason=type(generation_error).__name__,
                        )
            except AskTimeoutError as e:
                retrieval_error = e
                retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0
                log_ask_stage(
                    "error",
                    question=query,
                    elapsed_ms=retrieval_elapsed_ms,
                    reason=type(e).__name__,
                )
                st.session_state.last_retrieval_mode = "retrieval_failed"
                raw_results = []
                retrieved_docs = []
                context = ""
                st.session_state.last_query = query
                st.session_state.last_raw_results = []
                st.session_state.last_retrieved_docs = []
                st.session_state.last_context = ""
                st.session_state.last_answer = None
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as e:
                retrieval_error = e
                retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0
                log_ask_stage(
                    "error",
                    question=query,
                    elapsed_ms=retrieval_elapsed_ms,
                    reason=type(e).__name__,
                )
                st.session_state.last_retrieval_mode = "retrieval_failed"
                raw_results = []
                retrieved_docs = []
                context = ""
                st.session_state.last_query = query
                st.session_state.last_raw_results = []
                st.session_state.last_retrieved_docs = []
                st.session_state.last_context = ""
                st.session_state.last_answer = None
        finally:
            if timer_thread is not None:
                stop_timer.set()
                timer_thread.join(timeout=5.0)
            typing_slot.empty()
            timer_placeholder.empty()

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if retrieval_warning and st.session_state.developer_mode:
            st.caption(retrieval_warning)

        st.session_state.last_retrieval_metrics = {
            "strategy": st.session_state.retrieval_strategy,
            "mode": st.session_state.last_retrieval_mode,
            "reranker_enabled": st.session_state.enable_reranker,
            "retrieved_chunks": len(raw_results),
            "context_chars": len(context),
            "retrieval_ms": round(retrieval_elapsed_ms, 1),
            "rerank_ms": round(rerank_elapsed_ms, 1),
            "generation_ms": round(generation_elapsed_ms, 1),
            "total_ms": round(total_elapsed_ms, 1),
            **retrieval_diag,
        }

        if st.session_state.developer_mode:
            query_preview = _question_preview(query)
            if st.session_state.last_retrieval_metrics:
                with st.expander(
                    _dev_panel_title(
                        "Retrieval metrics (last run)", query_preview, fallback="live"
                    ),
                    expanded=False,
                ):
                    metrics = st.session_state.last_retrieval_metrics
                    st.markdown(
                        f"- strategy: `{metrics['strategy']}`  \n"
                        f"- mode: `{metrics['mode']}`  \n"
                        f"- reranker enabled: `{metrics['reranker_enabled']}`  \n"
                        f"- candidates (dense/sparse/fused): "
                        f"`{metrics.get('dense_candidates', 0)}` / "
                        f"`{metrics.get('sparse_candidates', 0)}` / "
                        f"`{metrics.get('fused_candidates', metrics.get('dense_candidates', 0))}`"
                        "  \n"
                        f"- selected before rerank: "
                        f"`{metrics.get('selected_before_rerank', 0)}`  \n"
                        f"- retrieved chunks: `{metrics['retrieved_chunks']}`  \n"
                        f"- context chars: `{metrics['context_chars']}`  \n"
                        f"- timing ms (retrieval/rerank/generation/total): "
                        f"`{metrics['retrieval_ms']}` / `{metrics['rerank_ms']}` / "
                        f"`{metrics['generation_ms']}` / `{metrics['total_ms']}`"
                    )
                    if retrieval_warning:
                        st.caption(f"Retrieval warning: {retrieval_warning}")
            with st.expander(
                _dev_panel_title("Exact context fed to LLM", query_preview, fallback="live"),
                expanded=False,
            ):
                st.code(context, language="text")
                st.caption(f"• {len(raw_results)} chunks • {len(context)} chars · top-k={TOP_K}")
            with st.expander(
                _dev_panel_title("Retrieved raw chunks + scores", query_preview, fallback="live"),
                expanded=False,
            ):
                for i, (doc, _raw_score) in enumerate(raw_results, start=1):
                    similarity = doc.metadata.get("similarity")
                    score = round(similarity, 3) if isinstance(similarity, (float, int)) else "N/A"
                    st.markdown(
                        f"**Chunk {i}** (similarity score: {score})  \n"
                        f"Page: {doc.metadata.get('page', '?')}"
                    )
                    st.code(doc.page_content, language="text")

    timing_payload = {
        "total_ms": total_elapsed_ms,
        "retrieval_ms": retrieval_elapsed_ms,
        "generation_ms": generation_elapsed_ms,
    }

    ready_name = st.session_state.current_file or "Document"
    if isinstance(retrieval_error, AskTimeoutError):
        generation_error = retrieval_error
        retrieval_error = None
    if retrieval_error is not None:
        retrieval_reason = (
            "I couldn't retrieve relevant document chunks right now. Please try again."
        )
        _remember_response_timing(timing_payload if total_elapsed_ms > 0 else None)
        _append_chat_message(
            "assistant",
            retrieval_reason,
            timing=timing_payload if total_elapsed_ms > 0 else None,
            for_question=query,
        )
        paint_status(status_slot, "error", retrieval_reason)
        st.rerun()
    elif generation_error is not None:
        e = generation_error
        st.session_state.last_answer = None
        if isinstance(e, AskTimeoutError):
            reason = _CLIENT_COPY["doc_timeout"]
        else:
            reason, _ = _generation_failure_reason(e)
        st.session_state.last_generation_error = {
            "user_reason": reason,
            "detail": str(e),
        }
        assistant_message = f"I couldn't generate an answer right now. {reason}"
        _remember_response_timing(timing_payload if total_elapsed_ms > 0 else None)
        _append_chat_message(
            "assistant",
            assistant_message,
            timing=timing_payload if total_elapsed_ms > 0 else None,
            for_question=query,
        )
        paint_status(status_slot, "error", reason)
        st.rerun()
    else:
        assistant_message = visitor_refusal_answer(
            st.session_state.last_answer
            or "I could not generate an answer at this time. Please try again.",
            query,
        )
        target_section = extract_target_section(query)
        source_payload = _build_sources_payload(
            retrieved_docs,
            query=query,
            answer=assistant_message,
            display_max=SOURCES_DISPLAY_MAX,
            preview_chars=SOURCE_PREVIEW_CHARS,
        )
        _remember_response_timing(timing_payload if total_elapsed_ms > 0 else None)
        _append_chat_message(
            "assistant",
            assistant_message,
            sources=source_payload,
            timing=timing_payload if total_elapsed_ms > 0 else None,
            for_question=query,
            eval_context=context if st.session_state.developer_mode else None,
            target_section=target_section,
            eval_chunk_count=len(raw_results),
        )
        paint_status(status_slot, "ready", _ready_message(ready_name))
        st.rerun()
elif query is not None and not chat_ready:
    paint_status(status_slot, "empty", _chat_blocked_user_message(uploaded_file))
elif query is not None:
    paint_status(status_slot, "error", "Please enter a non-empty question.")

# Newest answer first so the last reply stays on screen without iframe scroll.
turns = _paired_turns(st.session_state.messages)
indexed_stats = st.session_state.get("indexed_doc_stats") or {}
page_count = int(indexed_stats.get("text_pages") or indexed_stats.get("pages") or 0)
any_blank_pages = int(indexed_stats.get("blank_pages") or 0) > 0
pdf_bytes = st.session_state.get("uploaded_pdf_bytes")
assistant_count = sum(
    1 for message in st.session_state.messages if message.get("role") == "assistant"
)

for turn_idx, (user_msg, assistant_msg) in enumerate(reversed(turns)):
    question = (user_msg or {}).get("content") or assistant_msg.get("for_question") or ""
    _render_answer_card(
        question=question,
        answer=assistant_msg.get("content") or "",
        sources=assistant_msg.get("sources") or [],
        refusal=assistant_msg.get("refusal"),
        expanded=turn_idx == 0,
        timing=assistant_msg.get("timing"),
        developer_mode=st.session_state.developer_mode,
        page_count=page_count,
        any_blank_pages=any_blank_pages,
        pdf_bytes=pdf_bytes,
        card_key=str(len(turns) - turn_idx),
    )
    if turn_idx == 0 and assistant_count >= 3:
        _render_booking_cta(booking_url())
    if st.session_state.developer_mode and assistant_msg.get("sources"):
        turn_label = f"turn {len(turns) - turn_idx}"
        _render_source_checklist(
            assistant_msg["sources"],
            target_section=assistant_msg.get("target_section"),
            title=_dev_panel_title(
                "Source checklist (eval)",
                assistant_msg.get("for_question"),
                fallback=turn_label,
            ),
        )
        if assistant_msg.get("eval_context"):
            _render_eval_context_panel(
                assistant_msg["eval_context"],
                target_section=assistant_msg.get("target_section"),
                chunk_count=assistant_msg.get("eval_chunk_count"),
                title=_dev_panel_title(
                    "Exact context fed to LLM",
                    assistant_msg.get("for_question"),
                    fallback=turn_label,
                ),
                top_k=TOP_K,
            )

if not _show_sidebar:
    _render_client_footer(
        generator_label=_active_generator_label(st.session_state.dummy_generator_only),
        github_url=_github,
    )

if st.session_state.developer_mode:
    _render_persistent_generation_error()
