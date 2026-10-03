import streamlit as st
import streamlit.components.v1 as components
import tempfile
import threading
import yaml
import time
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from api_client import api_base_url, chat_via_api, chat_via_api_stream
from page_copy import _CLIENT_COPY, _render_client_hero, _render_sidebar_pitch
from session_upload import (
    _cache_upload,
    _chat_blocked_user_message,
    _chat_input_placeholder,
    _document_is_indexed,
    _init_session_state,
    _on_new_browser_session,
    _render_document_status,
    _reset_document_state,
    _resolve_upload,
    _set_dev_index_logs,
    _set_indexed_doc_stats,
    _set_processed_document,
)
from sources_panel import (
    _build_sources_payload,
    _dev_panel_title,
    _question_preview,
    _remember_response_timing,
    _render_answer_timing,
    _render_eval_context_panel,
    _render_source_checklist,
    _render_sources_panel,
    _sources_panel_title,
)
from rag import (
    generate_answer,
    generate_answer_stream,
    load_generation_config,
    resolve_llm_provider_name,
)
from rag.chunking import apply_hard_section_context_filter, chunk_pages, extract_target_section
from rag.citations import (
    INSUFFICIENT_CONTEXT_ANSWER,
    assemble_context,
    context_sufficient_for_query,
)
from rag.ingestion import extract_pdf
from rag.retrieval import (
    RetrievalConfig,
    build_bm25_index,
    finalize_retrieval,
    first_stage_retrieval,
)
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

st.set_page_config(
    page_title="Document Q&A · Private RAG",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)
APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=APP_ROOT / ".env")
MAX_CHAT_MESSAGES = 40


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


def _sample_nda_bytes() -> bytes | None:
    """Bytes for the walkthrough sample NDA, if present in the repo."""
    path = APP_ROOT / "docs" / "product" / "sample-nda.pdf"
    if not path.is_file():
        return None
    return path.read_bytes()


def _apply_presentation_mode_lock() -> None:
    """
    Pin UI mode from APP_PRESENTATION_MODE when the dev toggle is hidden.
    Prevents clients on VPS from seeing or enabling developer controls.
    """
    if _dev_toggle_allowed():
        return
    st.session_state.developer_mode = _presentation_mode() == "developer"


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
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Cache embedding model to avoid repeated heavy initialization."""
    return HuggingFaceEmbeddings(model_name=model_name)


def finalize_progress(progress_bar, message: str) -> None:
    """Set progress to done, pause briefly, then hide the bar."""
    if progress_bar is not None:
        progress_bar.progress(100, text=message)
        time.sleep(0.4)
        progress_bar.empty()



def _request_chat_scroll_to_bottom() -> None:
    """After a new assistant turn, scroll main pane to the latest message."""
    st.session_state._scroll_chat_to_bottom = True


def _scroll_chat_to_bottom_if_requested() -> None:
    if not st.session_state.pop("_scroll_chat_to_bottom", False):
        return
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const anchor = doc.getElementById("chat-scroll-anchor");
          if (anchor) {
            anchor.scrollIntoView({ behavior: "instant", block: "end" });
            return;
          }
          const main = doc.querySelector("section.main");
          if (main) {
            main.scrollTop = main.scrollHeight;
          }
        })();
        </script>
        """,
        height=0,
    )


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
    if role == "assistant":
        _request_chat_scroll_to_bottom()


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
    if diag.get("used_global_fallback"):
        st.info(
            f"No matches found in pages <= {EARLY_PAGE_MAX}. "
            "Showing best matches from all pages."
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
            f"(`OLLAMA_NUM_CTX` in `.env`). See `docs/ollama-troubleshooting.md`."
        )
        st.markdown("Run these commands in a terminal:")
        st.code(
            "ollama serve\n"
            f"{pull_commands}\n"
            "ollama list",
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
                    reason = "Anthropic error: " + detail.split("message': '", 1)[1].split("'", 1)[0]
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
    elif "mtllibrary" in error_text or "metal" in error_text or "llama runner process has terminated" in error_text:
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


def _stream_text_chunks(text: str, chunk_size: int = 40):
    """Yield text in small chunks for progressive chat rendering."""
    safe_text = text or ""
    for i in range(0, len(safe_text), chunk_size):
        yield safe_text[i:i + chunk_size]



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

    if ocr_pages_used > 0:
        st.warning(
            f"OCR used on {ocr_pages_used} page(s). Results may vary by scan quality."
        )
    elif scanned_pages_detected > 0 and ocr_warning:
        st.warning(ocr_warning)
        if developer_mode:
            st.caption(
                "Install OCR runtime first (`brew install tesseract`) and ensure "
                "`pytesseract` + `pillow` are available in the active environment."
            )



_init_session_state(
    developer_mode=_presentation_mode() == "developer",
    dummy_generator_only=_env_bool("USE_DUMMY_GENERATOR", False),
    retrieval_strategy_default=RETRIEVAL_STRATEGY_DEFAULT,
    reranker_enabled_default=RERANKER_ENABLED_DEFAULT,
)
_on_new_browser_session()
_apply_presentation_mode_lock()

_github = "https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline"

# Sidebar IA: Generator → GitHub → model → pitch/steps → sample → Exit → developer
st.sidebar.markdown("**Generator**")
st.sidebar.markdown(
    f"""
    <div class="app-sidebar-links app-sidebar-links--under-generator">
      <a class="app-sidebar-github"
         href="{_github}"
         target="_blank"
         rel="noopener noreferrer"
         aria-label="Project on GitHub"
         title="Project on GitHub">
        <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
          <path fill="currentColor"
            d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
            0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
            -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07
            -.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08
            -.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2
            .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
            0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0
            .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
        </svg>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption(_active_generator_label(st.session_state.dummy_generator_only))
if st.session_state.developer_mode:
    _dev_provider = resolve_llm_provider_name(st.session_state.dummy_generator_only)
    _dev_settings = load_generation_config()
    _dev_model = (
        _dev_settings.get("anthropic_model")
        if _dev_provider == "anthropic"
        else _dev_settings.get("model")
    )
    if _dev_model:
        st.sidebar.caption(f"API id: `{_dev_model}`")

_render_sidebar_pitch()

_sample_nda = _sample_nda_bytes()
if _sample_nda is not None:
    st.sidebar.download_button(
        label="Download sample NDA",
        data=_sample_nda,
        file_name="sample-nda.pdf",
        mime="application/pdf",
        use_container_width=True,
        help="Fictional sample for the walkthrough. Do not upload real company files on the shared pilot.",
    )

if st.session_state.developer_mode:
    st.sidebar.caption(
        "Stack: Python · LangChain · FAISS · Streamlit · OCR · Ollama"
    )

# Invite cookie clear → public gate. No-op locally without the invite proxy.
st.sidebar.link_button("Exit demo", "/invite/exit")

# Developer controls stay below client-facing sections
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
                "OFF: Ollama. Non-empty LLM_PROVIDER (e.g. anthropic) always wins — "
                "see Generator in the sidebar."
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

_render_client_hero()

uploader_key = f"pdf_uploader_{st.session_state.uploader_key_version}"
uploaded_file = st.file_uploader(
    "PDF",
    type=["pdf"],
    key=uploader_key,
    label_visibility="collapsed",
)

if st.session_state.pop("_fresh_session_hint", False) and not _document_is_indexed():
    st.info(_CLIENT_COPY["session_fresh"])

prev_uploader_had_file = st.session_state.get("_uploader_widget_had_file", False)
curr_uploader_has_file = uploaded_file is not None
if prev_uploader_had_file and not curr_uploader_has_file:
    if _document_is_indexed() or st.session_state.get("uploaded_pdf_bytes"):
        _reset_document_state(bump_uploader_key=True)
        st.session_state._uploader_widget_had_file = False
        st.info(_CLIENT_COPY["doc_cleared"])
        st.rerun()
st.session_state._uploader_widget_had_file = curr_uploader_has_file

extracted_text = ""

resolved_upload = _resolve_upload(uploaded_file)
if resolved_upload is not None:
    file_bytes, file_name = resolved_upload
    uploaded_hash = _cache_upload(file_bytes, file_name)

    if uploaded_file is None and not _document_is_indexed():
        st.info(_CLIENT_COPY["reindex_resume"])

    if (
        uploaded_hash == st.session_state.last_processed_hash
        and _document_is_indexed()
    ):
        st.session_state.current_file = file_name
        prev = st.session_state.indexed_doc_stats or {}
        _set_indexed_doc_stats(
            file_name=file_name,
            pages=int(prev.get("pages", 0)),
            chars=int(prev.get("chars", 0)),
            reused_index=True,
            header_split=SPLIT_ON_LEGAL_HEADERS,
            chunks=prev.get("chunks"),
        )
    else:
        tmp_path = None
        progress_bar = None
        try:
            progress_bar = st.progress(5, text=_CLIENT_COPY["progress_prepare"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = Path(tmp_file.name)

            progress_bar.progress(25, text=(
                "Extracting text from PDF…"
                if st.session_state.developer_mode
                else _CLIENT_COPY["progress_extract"]
            ))

            extracted = extract_pdf(tmp_path, enable_ocr=st.session_state.enable_ocr)
            page_docs = extracted.page_docs
            extracted_text = extracted.preview_text
            progress_bar.progress(45, text=(
                "Text extracted. Preparing chunking…"
                if st.session_state.developer_mode
                else _CLIENT_COPY["progress_chunk"]
            ))

            extracted_char_count = len(extracted_text.strip())
            if st.session_state.developer_mode:
                st.caption(
                    f"Extraction: **{len(page_docs)}** pages · **{extracted_char_count:,}** characters"
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

            with st.spinner(
                "Splitting text into chunks…"
                if st.session_state.developer_mode
                else _CLIENT_COPY["progress_chunk"]
            ):
                chunks = chunk_pages(
                    page_docs,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    split_on_legal_headers=SPLIT_ON_LEGAL_HEADERS,
                )
            progress_bar.progress(65, text=(
                "Chunks created. Loading embedding model…"
                if st.session_state.developer_mode
                else _CLIENT_COPY["progress_embed"]
            ))

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
                )
                finalize_progress(progress_bar, "No index created (no extractable text).")
                st.warning(
                    "No extractable text chunks were found in this document, "
                    "so search indexing was skipped. Try a PDF with selectable text "
                    "or run OCR preprocessing."
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
                progress_bar.progress(80, text=(
                    "Embeddings ready. Building FAISS index…"
                    if st.session_state.developer_mode
                    else _CLIENT_COPY["progress_embed"]
                ))
                with st.spinner(
                    f"Generating embeddings with {EMBEDDING_MODEL} & building FAISS index…"
                    if st.session_state.developer_mode
                    else _CLIENT_COPY["progress_index"]
                ):
                    start = time.time()

                    embeddings = get_embeddings(EMBEDDING_MODEL)

                    vector_store = FAISS.from_documents(
                        documents=st.session_state.chunks,
                        embedding=embeddings
                    )

                    st.session_state.vector_store = vector_store
                    took = time.time() - start
                finalize_progress(
                    progress_bar,
                    (
                        "Indexing complete."
                        if st.session_state.developer_mode
                        else _CLIENT_COPY["progress_done"]
                    ),
                )

                if st.session_state.developer_mode:
                    _set_dev_index_logs(
                        faiss_msg=(
                            f"FAISS index {'re-' if had_existing_index else ''}created "
                            f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
                        ),
                    )
                else:
                    st.toast(_CLIENT_COPY["toast_indexed"], icon="✅")

                _set_processed_document(file_name, uploaded_hash)
                _set_indexed_doc_stats(
                    file_name=file_name,
                    pages=len(page_docs),
                    chars=extracted_char_count,
                    reused_index=False,
                    header_split=SPLIT_ON_LEGAL_HEADERS,
                    chunks=len(chunks),
                )

                if len(chunks) <= 2:
                    st.warning("Very little text found in document. Search might not work well.")

        except (OSError, ValueError, RuntimeError) as e:
            finalize_progress(progress_bar, "Processing failed.")
            st.error(f"Error processing PDF: {str(e)}")
            st.exception(e)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

_render_document_status(
    uploaded_file=uploaded_file,
    resolved_upload=resolved_upload,
)

# Chat history UI (persists across reruns)
latest_sources_idx = next(
    (
        idx
        for idx in range(len(st.session_state.messages) - 1, -1, -1)
        if st.session_state.messages[idx].get("role") == "assistant"
        and st.session_state.messages[idx].get("sources")
    ),
    None,
)
for msg_idx, message in enumerate(st.session_state.messages):
    turn_label = f"turn {msg_idx // 2 + 1}"

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_answer_timing(
                message.get("timing"),
                developer_mode=st.session_state.developer_mode,
            )
            if message.get("sources"):
                _render_sources_panel(
                    message["sources"],
                    developer_mode=st.session_state.developer_mode,
                    title=_sources_panel_title(message),
                    expanded=msg_idx == latest_sources_idx,
                )
            if st.session_state.developer_mode and message.get("sources"):
                _render_source_checklist(
                    message["sources"],
                    target_section=message.get("target_section"),
                    title=_dev_panel_title(
                        "Source checklist (eval)",
                        message.get("for_question"),
                        fallback=turn_label,
                    ),
                )
                if message.get("eval_context"):
                    _render_eval_context_panel(
                        message["eval_context"],
                        target_section=message.get("target_section"),
                        chunk_count=message.get("eval_chunk_count"),
                        title=_dev_panel_title(
                            "Exact context fed to LLM",
                            message.get("for_question"),
                            fallback=turn_label,
                        ),
                        top_k=TOP_K,
                    )

st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)
_scroll_chat_to_bottom_if_requested()

chat_ready = _document_is_indexed()
query = st.chat_input(
    _chat_input_placeholder(
        uploaded_file=uploaded_file,
        resolved_upload=resolved_upload,
    ),
    disabled=not chat_ready,
)

if query and query.strip() and chat_ready:
    query = query.strip()
    _append_chat_message("user", query)

    with st.chat_message("user"):
        st.markdown(query)

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

    with st.chat_message("assistant"):
        timer_placeholder = st.empty()
        stop_timer = threading.Event()
        script_ctx = get_script_run_ctx()
        need_stream_generation = False
        total_start = time.perf_counter()

        def _thinking_timer_loop() -> None:
            if script_ctx is not None:
                add_script_run_ctx(threading.current_thread(), script_ctx)
            t0 = time.perf_counter()
            while not stop_timer.wait(0.12):
                elapsed = time.perf_counter() - t0
                timer_placeholder.caption(f"Working… {elapsed:.1f}s")

        with st.spinner("Working on your answer…"):
            timer_thread = None
            if script_ctx is not None:
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
                        max(TOP_K, RERANKER_TOP_N)
                        if st.session_state.enable_reranker
                        else TOP_K
                    )
                    candidate_pool, mode, first_stage_warning, retrieval_diag = (
                        _run_first_stage_retrieval(
                            query,
                            candidate_limit=candidate_limit,
                        )
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
                    retrieval_elapsed_ms = (
                        time.perf_counter() - retrieval_start
                    ) * 1000.0

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
                            part
                            for part in (retrieval_warning, hard_filter_warning)
                            if part
                        )
                        st.session_state.last_retrieval_mode = (
                            f"{st.session_state.last_retrieval_mode}_section_filtered"
                        )

                    retrieved_docs = [doc for doc, _score in context_results]
                    st.session_state.last_query = query
                    st.session_state.last_retrieved_docs = retrieved_docs
                    context = assemble_context(
                        context_results,
                        use_page_separators=st.session_state.use_page_separators,
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
                        need_stream_generation = True
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ) as e:
                    retrieval_error = e
                    retrieval_elapsed_ms = (
                        time.perf_counter() - retrieval_start
                    ) * 1000.0
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
        timer_placeholder.empty()

        # Stream tokens after retrieval so the UI feels responsive; fall back if needed.
        # When API_BASE_URL is set, generation goes through the thin FastAPI /chat.
        if need_stream_generation:
            generation_start = time.perf_counter()
            dummy_mode = st.session_state.dummy_generator_only
            use_api = bool(api_base_url())
            try:
                try:
                    stream_fn = (
                        chat_via_api_stream
                        if use_api
                        else generate_answer_stream
                    )
                    streamed = st.write_stream(
                        stream_fn(
                            context=context,
                            query=query,
                            dummy_mode=dummy_mode,
                        )
                    )
                except Exception:
                    full_answer = (
                        chat_via_api(
                            context=context,
                            query=query,
                            dummy_mode=dummy_mode,
                        )
                        if use_api
                        else generate_answer(
                            context=context,
                            query=query,
                            dummy_mode=dummy_mode,
                        )
                    )
                    streamed = st.write_stream(_stream_text_chunks(full_answer))
                    if not streamed:
                        streamed = full_answer
                        st.markdown(full_answer)
                st.session_state.last_answer = streamed if isinstance(streamed, str) else None
                st.session_state.last_generation_error = None
            except Exception as e:
                generation_error = e
                st.session_state.last_answer = None
            generation_elapsed_ms = (
                time.perf_counter() - generation_start
            ) * 1000.0

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if retrieval_warning:
            st.warning(retrieval_warning)

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
                    _dev_panel_title("Retrieval metrics (last run)", query_preview, fallback="live"),
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
                        f"`{metrics.get('fused_candidates', metrics.get('dense_candidates', 0))}`  \n"
                        f"- selected before rerank: `{metrics.get('selected_before_rerank', 0)}`  \n"
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
                    st.markdown(
                        f"**Chunk {i}** (similarity score: {round(similarity, 3) if isinstance(similarity, (float, int)) else 'N/A'})  \n"
                        f"Page: {doc.metadata.get('page', '?')}"
                    )
                    st.code(doc.page_content, language="text")

    timing_payload = {
        "total_ms": total_elapsed_ms,
        "retrieval_ms": retrieval_elapsed_ms,
        "generation_ms": generation_elapsed_ms,
    }

    if retrieval_error is not None:
        retrieval_reason = (
            "I couldn't retrieve relevant document chunks right now. "
            "Please try again."
        )
        _remember_response_timing(timing_payload if total_elapsed_ms > 0 else None)
        _append_chat_message(
            "assistant",
            retrieval_reason,
            timing=timing_payload if total_elapsed_ms > 0 else None,
            for_question=query,
        )
        st.rerun()
    elif generation_error is not None:
        e = generation_error
        st.session_state.last_answer = None
        reason, _ = _generation_failure_reason(e)
        st.session_state.last_generation_error = {
            "user_reason": reason,
            "detail": str(e),
        }
        assistant_message = (
            "I couldn't generate an answer right now. "
            f"{reason}"
        )
        _remember_response_timing(timing_payload if total_elapsed_ms > 0 else None)
        _append_chat_message(
            "assistant",
            assistant_message,
            timing=timing_payload if total_elapsed_ms > 0 else None,
            for_question=query,
        )
        st.rerun()
    else:
        assistant_message = (
            st.session_state.last_answer
            or "I could not generate an answer at this time. Please try again."
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
        st.rerun()
elif query is not None and not chat_ready:
    st.info(_chat_blocked_user_message(uploaded_file))
elif query is not None:
    st.warning("Please enter a non-empty question.")

_render_persistent_generation_error()
