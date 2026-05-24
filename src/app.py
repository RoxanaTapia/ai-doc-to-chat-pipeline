import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF (pymupdf package)
import tempfile
import threading
import yaml
import time
import hashlib
import os
import io
import re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports for Milestone 3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag import generate_answer, load_generation_config
from retrieval_quality import (
    INSUFFICIENT_CONTEXT_ANSWER,
    context_sufficient_for_query,
    dedupe_similar_chunks,
)
from sectioning import (
    annotate_chunk_sections,
    apply_hard_section_context_filter,
    apply_section_aware_retrieval,
    split_documents_by_legal_headers,
    limit_chunks_per_page,
    chunk_on_section,
    extract_target_section,
)
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

st.set_page_config(page_title="Ask your PDF · Private RAG", layout="wide")
APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=APP_ROOT / ".env")
MAX_CHAT_MESSAGES = 40
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Stable for the lifetime of the Streamlit process (survives reruns, changes on container restart).
_APP_BOOT_ID = str(os.getpid())

# Client-demo microcopy — memorable pilot tone, serious for legal/ops buyers
_CLIENT_COPY = {
    "hero_title": "🔒 Ask your PDF anything ✨",
    "hero_lead": (
        "Private, grounded answers with <strong>page-level sources</strong> — "
        "on infrastructure you control."
    ),
    "hero_kicker": "Upload · index · ask. No public ChatGPT tab required.",
    "chat_ready": "What would you like to know?",
    "chat_indexing": "Almost there…",
    "chat_waiting": "Your questions land here once the PDF is ready.",
    "doc_ready": (
        "**{name}** is ready — your turn. Ask in plain language; "
        "open **Sources** under each answer to verify the page."
    ),
    "doc_indexing": "Getting to know **{name}**…",
    "doc_cleared": "Document removed — upload another PDF whenever you're ready.",
    "session_fresh": (
        "Upload a PDF below — when you see the green **ready** message, "
        "you can ask your first question."
    ),
    "doc_stale": (
        "This PDF is not indexed yet — wait for the green **ready** message, "
        "or use the **✕** on the file above and upload again."
    ),
}


def _is_probably_streamlit_cloud() -> bool:
    """Best-effort detection for hosted Streamlit runtimes."""
    return any(
        os.getenv(marker)
        for marker in ("STREAMLIT_SHARING_MODE", "STREAMLIT_RUNTIME", "STREAMLIT_CLOUD")
    )


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
    """Whether the sidebar shows the developer-mode toggle."""
    return _env_bool(
        "APP_ALLOW_DEV_TOGGLE",
        not _is_probably_streamlit_cloud(),
    )


def _inject_demo_styles() -> None:
    """Light hero styling for client-facing demos (Streamlit-safe CSS)."""
    st.markdown(
        """
        <style>
        .app-hero { margin: -0.75rem 0 1.25rem 0; }
        .app-hero__lead {
            font-size: 1.15rem;
            line-height: 1.55;
            color: #4b5563;
            margin: 0 0 0.45rem 0;
        }
        .app-hero__kicker {
            font-size: 0.95rem;
            color: #9ca3af;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_client_hero() -> None:
    """Main headline — trustworthy, memorable, hire-me without being salesy."""
    _inject_demo_styles()
    st.title(_CLIENT_COPY["hero_title"])
    st.markdown(
        f"""
        <div class="app-hero">
          <p class="app-hero__lead">{_CLIENT_COPY["hero_lead"]}</p>
          <p class="app-hero__kicker">{_CLIENT_COPY["hero_kicker"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _on_new_browser_session() -> None:
    """
    New Streamlit session (tab refresh, reconnect after compose restart).
    Bump uploader key so ghost filenames do not look 'ready' without an index.
    """
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


def _apply_presentation_mode_lock() -> None:
    """
    When the dev toggle is hidden, pin UI mode from APP_PRESENTATION_MODE.
    Prevents clients on VPS from seeing or enabling developer controls.
    """
    if not _dev_toggle_allowed():
        st.session_state.developer_mode = _presentation_mode() == "developer"


def _upload_in_flight(*, uploaded_file, resolved_upload: tuple[bytes, str] | None) -> bool:
    """True when we have file bytes but no searchable index yet."""
    if _document_is_indexed():
        return False
    return resolved_upload is not None or uploaded_file is not None or bool(
        st.session_state.get("uploaded_pdf_bytes")
    )


if _is_probably_streamlit_cloud():
    st.sidebar.info(
        "**Public demo (Streamlit Cloud)** — uploads here are temporary. "
        "**Answers use the dummy generator only** (no Ollama on this host). "
        "For **real private generation** with your PDFs, run the app **locally** with Ollama, "
        "or **request a live session** if you need a hosted setup with a real model."
    )

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

    ui_cfg = rag_cfg.get("ui", {}) or {}
    SOURCES_DISPLAY_MAX = int(ui_cfg.get("sources_display_max", 5))
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


def _readable_excerpt(text: str, max_chars: int) -> str:
    """Trim chunk text at word or sentence boundaries for human-readable source previews."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    snippet = cleaned[:max_chars]
    for sep in (". ", "; ", ", ", " "):
        cut = snippet.rfind(sep)
        if cut > max_chars // 2:
            return f"{snippet[: cut + len(sep)].strip()}…"
    return f"{snippet.rstrip()}…"


EVAL_CHECKLIST_PREVIEW_CHARS = 80


def _build_sources_payload(
    retrieved_docs: list[Document],
    *,
    query: str | None = None,
) -> list[dict]:
    """Create a compact serializable source payload per assistant answer."""
    target_section = extract_target_section(query) if query else None
    payload = []
    for chunk in retrieved_docs[:SOURCES_DISPLAY_MAX]:
        similarity = chunk.metadata.get("similarity")
        entry = {
            "page": chunk.metadata.get("page", "N/A"),
            "score": round(similarity, 3) if isinstance(similarity, (float, int)) else "N/A",
            "preview": _readable_excerpt(chunk.page_content, SOURCE_PREVIEW_CHARS),
        }
        if target_section is not None:
            entry["checklist_preview"] = _readable_excerpt(
                chunk.page_content,
                EVAL_CHECKLIST_PREVIEW_CHARS,
            )
            entry["on_section"] = chunk_on_section(chunk, target_section)
        payload.append(entry)
    return payload


def _dev_panel_title(base: str, question_preview: str | None, *, fallback: str) -> str:
    """Unique expander label per turn (Streamlit 1.54 has no expander key=)."""
    preview = (question_preview or "").strip()
    if preview:
        return f"{base} — {_question_preview(preview, 40)}"
    return f"{base} — {fallback}"


def _render_sources_panel(
    sources: list[dict],
    *,
    developer_mode: bool,
    title: str = "Sources",
) -> None:
    """Render source excerpts in a client-friendly expander."""
    if not sources:
        return
    with st.expander(title, expanded=False):
        for source_idx, source in enumerate(sources, start=1):
            header = f"**Source {source_idx}** · Page {source['page']}"
            if developer_mode and source.get("score") != "N/A":
                header += f" · relevance {source['score']}"
            st.markdown(header)
            st.markdown(f"> {source['preview']}")


def _on_section_label(on_section: bool | None) -> str:
    if on_section is True:
        return "Yes"
    if on_section is False:
        return "No"
    return "N/A"


def _render_source_checklist(
    sources: list[dict],
    *,
    target_section: str | None = None,
    title: str = "📋 Source checklist (eval)",
) -> None:
    """Dev-only Round 4 eval aid: page, ~80 char excerpt, on-section Y/N per source."""
    if not sources:
        return
    with st.expander(title, expanded=False):
        if target_section:
            st.caption(
                f"Target section: **{target_section}** · auto-tag is heuristic — "
                "confirm manually when scoring."
            )
        else:
            st.caption("No section in question — on-section column shows N/A.")
        for source_idx, source in enumerate(sources, start=1):
            preview = source.get("checklist_preview") or source.get("preview", "")
            st.markdown(
                f"{source_idx}. **Page {source['page']}** · "
                f"on-section {_on_section_label(source.get('on_section'))}  \n"
                f"> {preview}"
            )
        if target_section:
            on_count = sum(1 for source in sources if source.get("on_section") is True)
            st.caption(
                f"Auto on-section count: **{on_count}/{len(sources)}** "
                "(Round 4 bar: ≥3/5 on-section for Q1 and Q2)."
            )


def _render_eval_context_panel(
    eval_context: str,
    *,
    target_section: str | None = None,
    chunk_count: int | None = None,
    title: str = "📄 Exact Context fed to LLM",
) -> None:
    """Dev-only: persisted exact context for retrieval vs generation diagnosis."""
    with st.expander(title, expanded=False):
        st.code(eval_context, language="text")
        chunk_note = f" • {chunk_count} chunks" if chunk_count else ""
        st.caption(f"• {len(eval_context)} chars{chunk_note} · top-k={TOP_K}")
        if target_section == "3":
            st.info(
                "Round 4 Q2 diagnostic: if return/destroy or termination language "
                "appears **in this context**, retrieval is likely at fault; "
                "if Section 3 duties are here but missing from the answer, "
                "generation is likely at fault."
            )


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


def _assemble_context(raw_results: list[tuple[Document, float]], use_page_separators: bool) -> str:
    """Build the exact context string that is fed to the generator."""
    if use_page_separators:
        chunks_with_pages = []
        for doc, _score in raw_results:
            page = doc.metadata.get("page", "?")
            chunks_with_pages.append(f"─── Page {page} ───\nPage {page}: {doc.page_content}")
        return "\n".join(chunks_with_pages)
    return "\n\n".join(doc.page_content for doc, _score in raw_results)


def _distance_to_ui_similarity(distance: float) -> float:
    """
    Convert vector distance to a bounded [0,1] similarity for UI display.
    Lower distance -> higher similarity.
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize arbitrary ranking scores to [0,1] for consistent UI display."""
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    minimum = min(scores)
    maximum = max(scores)
    if maximum - minimum <= 1e-12:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple lexical tokenizer used by BM25 ranking."""
    return TOKEN_RE.findall((text or "").lower())


def _doc_key(doc: Document) -> tuple:
    """Stable key for merging dense and sparse retrieval results."""
    stable_fingerprint = hashlib.blake2b(
        doc.page_content[:300].encode("utf-8", errors="ignore"),
        digest_size=12,
    ).hexdigest()
    return (
        doc.metadata.get("page", "N/A"),
        doc.metadata.get("start_index", "N/A"),
        stable_fingerprint,
    )


def _ensure_bm25_index() -> tuple[object | None, str | None]:
    """Build and cache BM25 index for current chunk set."""
    if st.session_state.get("bm25_state") is not None:
        state = st.session_state.bm25_state
        if state.get("chunk_count") == len(st.session_state.chunks or []):
            return state.get("index"), None

    chunks = st.session_state.chunks or []
    if not chunks:
        return None, "No chunks available for BM25 indexing."

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, (
            "Hybrid retrieval requires `rank-bm25`. "
            "Install it with `pip install -r requirements.txt`."
        )

    tokenized_corpus = [_tokenize_for_bm25(doc.page_content) for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    st.session_state.bm25_state = {
        "index": bm25_index,
        "tokenized_corpus": tokenized_corpus,
        "chunk_count": len(chunks),
    }
    return bm25_index, None


@st.cache_resource(show_spinner=False)
def _get_cross_encoder(model_name: str):
    """Cache cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _apply_reranker(
    query: str,
    candidates: list[tuple[Document, float]],
    *,
    top_n: int,
    model_name: str,
) -> tuple[list[tuple[Document, float]], str | None]:
    """Second-stage reranking with cross-encoder on a candidate pool."""
    if not candidates:
        return [], None
    try:
        reranker = _get_cross_encoder(model_name)
        effective_top_n = max(1, int(top_n))
        limited_candidates = candidates[:effective_top_n]
        pairs = [(query, doc.page_content[:2000]) for doc, _score in limited_candidates]
        raw_scores = reranker.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            score_values = [float(score) for score in raw_scores.tolist()]
        else:
            score_values = [float(score) for score in raw_scores]
        normalized = _normalize_scores(score_values)
        reranked = [
            (doc, normalized_score)
            for (doc, _original_score), normalized_score in zip(limited_candidates, normalized)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)

        final_results = reranked[:TOP_K]
        if len(final_results) < TOP_K:
            seen_keys = {_doc_key(doc) for doc, _score in final_results}
            for doc, score in candidates:
                key = _doc_key(doc)
                if key in seen_keys:
                    continue
                final_results.append((doc, score))
                seen_keys.add(key)
                if len(final_results) >= TOP_K:
                    break
        return final_results, None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return candidates[:TOP_K], (
            f"Reranker unavailable ({exc}). Falling back to first-stage ranking."
        )


def _semantic_retrieval(
    query: str,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[Document, float]], str, dict[str, int]]:
    """Dense retrieval with early-page preference."""
    effective_limit = max(1, int(limit or TOP_K))
    candidate_results = st.session_state.vector_store.similarity_search_with_score(
        query,
        k=FETCH_K,
    )

    early_page_results = []
    for doc, score in candidate_results:
        page = doc.metadata.get("page")
        if isinstance(page, int) and page <= EARLY_PAGE_MAX:
            early_page_results.append((doc, score))

    if early_page_results:
        selected = early_page_results[:effective_limit]
        mode = "semantic_early_page"
    else:
        selected = candidate_results[:effective_limit]
        mode = "semantic_global_fallback"
        st.info(
            f"No matches found in pages <= {EARLY_PAGE_MAX}. "
            "Showing best matches from all pages."
        )
    return [
        (doc, _distance_to_ui_similarity(float(distance)))
        for doc, distance in selected
    ], mode, {
        "dense_candidates": len(candidate_results),
        "early_page_candidates": len(early_page_results),
        "selected_before_rerank": len(selected),
    }


def _hybrid_rrf_retrieval(
    query: str,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[Document, float]], str, str | None, dict[str, int]]:
    """Hybrid retrieval using dense + BM25 with Reciprocal Rank Fusion."""
    effective_limit = max(1, int(limit or TOP_K))
    dense_results = st.session_state.vector_store.similarity_search_with_score(
        query,
        k=FETCH_K,
    )
    bm25_index, bm25_warning = _ensure_bm25_index()
    if bm25_index is None:
        semantic_results, semantic_mode, semantic_diag = _semantic_retrieval(query, limit=effective_limit)
        hybrid_diag = {
            **semantic_diag,
            "sparse_candidates": 0,
            "fused_candidates": len(semantic_results),
        }
        return semantic_results, f"{semantic_mode}_bm25_unavailable", bm25_warning, hybrid_diag

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        semantic_results, semantic_mode, semantic_diag = _semantic_retrieval(query, limit=effective_limit)
        hybrid_diag = {
            **semantic_diag,
            "sparse_candidates": 0,
            "fused_candidates": len(semantic_results),
        }
        return semantic_results, f"{semantic_mode}_empty_query_tokens", None, hybrid_diag

    bm25_scores = bm25_index.get_scores(query_tokens)
    top_sparse_indices = [
        idx
        for idx, _score in sorted(
            enumerate(bm25_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:BM25_FETCH_K]
    ]

    fused_scores: dict[tuple, float] = defaultdict(float)
    doc_map: dict[tuple, Document] = {}

    for rank, (doc, _distance) in enumerate(dense_results, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        fused_scores[key] += DENSE_WEIGHT / (RRF_K + rank)

    for rank, chunk_idx in enumerate(top_sparse_indices, start=1):
        doc = st.session_state.chunks[chunk_idx]
        key = _doc_key(doc)
        doc_map[key] = doc
        fused_scores[key] += BM25_WEIGHT / (RRF_K + rank)

    ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
    top_candidates = [(doc_map[key], fused_scores[key]) for key in ranked_keys[:effective_limit]]
    normalized_scores = _normalize_scores([score for _doc, score in top_candidates])
    normalized_candidates = [
        (doc, normalized_score)
        for (doc, _original_score), normalized_score in zip(top_candidates, normalized_scores)
    ]
    return normalized_candidates[:effective_limit], "hybrid_rrf", bm25_warning, {
        "dense_candidates": len(dense_results),
        "sparse_candidates": len(top_sparse_indices),
        "fused_candidates": len(ranked_keys),
        "selected_before_rerank": len(top_candidates),
    }


def _format_elapsed_ms(elapsed_ms: float) -> str:
    """Format milliseconds as a short human-readable duration."""
    seconds = max(0.0, elapsed_ms / 1000.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(round(seconds % 60))
    if rem == 60:
        minutes += 1
        rem = 0
    return f"{minutes}m {rem}s" if rem else f"{minutes}m"


def _response_timing_caption(
    *,
    total_ms: float,
    retrieval_ms: float = 0.0,
    generation_ms: float = 0.0,
    developer_mode: bool = False,
) -> str:
    """Build a caption showing how long the last answer took."""
    total_label = _format_elapsed_ms(total_ms)
    if not developer_mode:
        return f"Answered in {total_label}"
    parts = []
    if retrieval_ms > 0:
        parts.append(f"retrieval {_format_elapsed_ms(retrieval_ms)}")
    if generation_ms > 0:
        parts.append(f"generation {_format_elapsed_ms(generation_ms)}")
    if parts:
        return f"Answered in {total_label} ({' · '.join(parts)})"
    return f"Answered in {total_label}"


def _assistant_message_with_timing(
    content: str,
    timing: dict[str, float] | None,
    *,
    developer_mode: bool,
) -> str:
    """Embed elapsed time in assistant markdown so it survives Streamlit reruns."""
    if not timing or timing.get("total_ms", 0) <= 0:
        return content
    footer = _response_timing_caption(
        total_ms=timing["total_ms"],
        retrieval_ms=timing.get("retrieval_ms", 0.0),
        generation_ms=timing.get("generation_ms", 0.0),
        developer_mode=developer_mode,
    )
    return f"{content}\n\n---\n*{footer}*"


def _remember_response_timing(timing: dict[str, float] | None) -> None:
    """Persist last response timing for the status line above chat input."""
    if timing and timing.get("total_ms", 0) > 0:
        st.session_state.last_response_timing = timing


def _question_preview(text: str, max_len: int = 52) -> str:
    """Short label for Sources expander — ties excerpts to the question asked."""
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1].rstrip() + "…"


def _sources_panel_title(message: dict) -> str:
    preview = message.get("for_question")
    if preview:
        return f"Sources — {preview}"
    return "Sources for this answer"


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
    if role == "assistant":
        content = _assistant_message_with_timing(
            content,
            timing,
            developer_mode=st.session_state.developer_mode,
        )
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
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


def _run_first_stage_retrieval(
    query: str,
    *,
    candidate_limit: int,
) -> tuple[list[tuple[Document, float]], str, str | None, dict[str, int]]:
    """Run the configured first-stage retrieval and return normalized candidates."""
    if st.session_state.retrieval_strategy == "hybrid":
        return _hybrid_rrf_retrieval(query, limit=candidate_limit)

    semantic_results, mode, semantic_diag = _semantic_retrieval(query, limit=candidate_limit)
    return semantic_results, mode, None, semantic_diag


def _finalize_retrieval_candidates(
    query: str,
    *,
    candidate_pool: list[tuple[Document, float]],
    mode: str,
) -> tuple[list[tuple[Document, float]], str, str | None, float]:
    """Apply optional reranker and section-aware routing; return final top-k results."""
    reranker_warning: str | None = None
    rerank_elapsed_ms = 0.0
    pool = candidate_pool

    if st.session_state.enable_reranker:
        rerank_start = time.perf_counter()
        pool, reranker_warning = _apply_reranker(
            query,
            candidate_pool,
            top_n=RERANKER_TOP_N,
            model_name=RERANKER_MODEL_NAME,
        )
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000.0
        mode = f"{mode}_reranked"

    section_warning: str | None = None
    if SECTION_AWARE_ENABLED:
        pool, section_warning = apply_section_aware_retrieval(
            query,
            pool,
            top_k=TOP_K,
            boost=SECTION_AWARE_BOOST,
            min_matching=SECTION_AWARE_MIN_CHUNKS,
        )
        if section_warning:
            mode = f"{mode}_section_aware"
    else:
        pool = pool[:TOP_K]

    if MAX_CHUNKS_PER_PAGE > 0:
        pool = limit_chunks_per_page(pool, top_k=TOP_K, max_per_page=MAX_CHUNKS_PER_PAGE)

    if DEDUPE_SIMILAR_CHUNKS:
        pool = dedupe_similar_chunks(
            pool,
            top_k=TOP_K,
            prefix_chars=DEDUPE_PREFIX_CHARS,
        )
        mode = f"{mode}_deduped"

    combined_warning = " · ".join(
        part for part in (reranker_warning, section_warning) if part
    ) or None
    return pool[:TOP_K], mode, combined_warning, rerank_elapsed_ms


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


def _generation_failure_reason(exc: BaseException) -> tuple[str, str]:
    """Return (short_user_reason, normalized_error_text)."""
    try:
        model_name = load_generation_config().get("model", "llama3.1:8b")
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
        model_name = "llama3.1:8b"
    error_text = str(exc).lower()
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
    _ollama_recovery_expander(expanded=False)
    detail = err.get("detail")
    if st.session_state.developer_mode and detail:
        with st.expander("Technical details", expanded=False):
            st.code(detail, language="text")
    if st.button("Dismiss", key="dismiss_generation_error_banner"):
        st.session_state.last_generation_error = None
        st.rerun()


def _set_indexed_doc_stats(
    *,
    file_name: str,
    pages: int,
    chars: int,
    reused_index: bool,
    chunks: int | None = None,
) -> None:
    st.session_state.indexed_doc_stats = {
        "file_name": file_name,
        "pages": pages,
        "chars": chars,
        "reused_index": reused_index,
        "chunks": chunks,
        "header_split": SPLIT_ON_LEGAL_HEADERS,
    }


def _set_dev_index_logs(
    *,
    chunks_msg: str | None = None,
    faiss_msg: str | None = None,
) -> None:
    """Persist dev indexing messages so they render above chat, not mid-turn."""
    logs = dict(st.session_state.get("dev_index_logs") or {})
    if chunks_msg is not None:
        logs["chunks"] = chunks_msg
    if faiss_msg is not None:
        logs["faiss"] = faiss_msg
    st.session_state.dev_index_logs = logs


def _clear_dev_index_logs() -> None:
    st.session_state.pop("dev_index_logs", None)


def _clear_cached_upload() -> None:
    """Drop persisted upload bytes (e.g. on clear document)."""
    for key in ("uploaded_pdf_bytes", "uploaded_pdf_name", "uploaded_pdf_hash"):
        st.session_state.pop(key, None)


def _cache_upload(file_bytes: bytes, file_name: str) -> str:
    """Persist upload in session so Rerun does not lose the file buffer."""
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    st.session_state.uploaded_pdf_bytes = file_bytes
    st.session_state.uploaded_pdf_name = file_name
    st.session_state.uploaded_pdf_hash = file_hash
    return file_hash


def _resolve_upload(uploaded_file) -> tuple[bytes, str] | None:
    """Return file bytes and name from the widget or session cache."""
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    cached_name = st.session_state.get("uploaded_pdf_name")
    cached_bytes = st.session_state.get("uploaded_pdf_bytes")
    if cached_name and cached_bytes:
        return cached_bytes, cached_name
    return None


def _document_is_indexed() -> bool:
    return st.session_state.vector_store is not None


def _chat_input_placeholder(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
) -> str:
    if _document_is_indexed():
        return _CLIENT_COPY["chat_ready"]
    if _upload_in_flight(uploaded_file=uploaded_file, resolved_upload=resolved_upload):
        return _CLIENT_COPY["chat_indexing"]
    return _CLIENT_COPY["chat_waiting"]


def _chat_blocked_user_message(uploaded_file) -> str:
    if _document_is_indexed():
        return ""
    if uploaded_file is not None or st.session_state.get("uploaded_pdf_bytes"):
        return (
            "Still indexing — wait for the green **ready** message, then ask again."
        )
    if st.session_state.current_file or st.session_state.get("uploaded_pdf_name"):
        return (
            "The file box may still show a name after **Rerun**, but the upload buffer was cleared. "
            "Use the **✕** on the PDF above, or upload again."
        )
    return "Upload a PDF and wait for indexing to finish before asking a question."


def _render_document_status(
    *,
    uploaded_file,
    resolved_upload: tuple[bytes, str] | None,
) -> None:
    """Show clear indexing / ready state — persists when toggling presentation mode."""
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


def _is_likely_scanned_page(text: str) -> bool:
    """Heuristic: very short extracted text suggests an image-only page."""
    return len((text or "").strip()) < 50


def _ocr_page_text(page) -> tuple[str, str | None]:
    """
    Try OCR on a page image.
    Returns (ocr_text, error_message). Error message is None on success.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", "OCR dependencies not installed (requires pytesseract + pillow + system tesseract)."

    try:
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        # Minimal preprocessing: simple thresholding helps many scanned contracts.
        image = image.point(lambda x: 0 if x < 180 else 255, mode="1")
        ocr_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6").strip()
        return ocr_text, None
    except (OSError, RuntimeError, ValueError) as exc:
        return "", f"OCR failed on page: {exc}"


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


def _reset_chat_state() -> None:
    """Reset query/answer history while keeping indexed document."""
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
    """Reset the indexed document and all retrieval/generation state."""
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
    """Persist document identity and reset chat history for a fresh index."""
    st.session_state.last_processed_name = file_name
    st.session_state.last_processed_hash = file_hash
    st.session_state.current_file = file_name
    _reset_chat_state()


def _init_session_state() -> None:
    """Initialize all required Streamlit session keys once."""
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
        "developer_mode": _presentation_mode() == "developer",
        "enable_ocr": True,
        "use_page_separators": True,
        # Local: Ollama by default. Cloud: dummy only (no local Ollama). Env overrides either way.
        "dummy_generator_only": _env_bool(
            "USE_DUMMY_GENERATOR",
            _is_probably_streamlit_cloud(),
        ),
        "bm25_state": None,
        "retrieval_strategy": (
            RETRIEVAL_STRATEGY_DEFAULT
            if RETRIEVAL_STRATEGY_DEFAULT in {"semantic", "hybrid"}
            else "semantic"
        ),
        "enable_reranker": RERANKER_ENABLED_DEFAULT,
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


_init_session_state()
_on_new_browser_session()
_apply_presentation_mode_lock()

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
with st.sidebar.expander("About", expanded=False):
    st.markdown(
        """
        **Built for teams who won't paste contracts into ChatGPT.**

        Upload a PDF, ask in plain language, get answers with **sources you
        can verify**. Everything runs on **your** infrastructure — private
        by design.

        Digital PDFs shine; scanned pages work too (OCR when needed).

        **Like what you see?** [Upwork](https://www.upwork.com/freelancers/roxanadev) ·
        [GitHub](https://github.com/RoxanaTapia)
        """
    )
    if st.session_state.developer_mode:
        st.caption(
            "Stack: Python · LangChain · FAISS · Streamlit · OCR · Ollama"
        )

with st.sidebar.expander("How to use", expanded=False):
    st.markdown(
        """
        **Three moves:**
        1. Drop a PDF — contract, policy, report.
        2. Wait for the green **ready** message.
        3. Ask like you'd ask a colleague — then open **Sources**.

        Name the section in your question for sharper answers.
        Scanned pages? OCR kicks in automatically.
        """
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
            help="ON: echo mode answer. OFF: try local Ollama and fallback if unavailable.",
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

if _is_probably_streamlit_cloud():
    st.warning(
        "**Demo mode:** this hosted app uses **dummy chat responses** only. "
        "There is no Ollama (or other LLM) on Streamlit Cloud. "
        "For **real answers on your documents**, run **`streamlit run src/app.py` locally** with Ollama installed, "
        "or **request a live session** for a private hosted demo."
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
        st.info("Welcome back — re-indexing your document now.")

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
            chunks=prev.get("chunks"),
        )
    else:
        tmp_path = None
        progress_bar = None
        try:
            progress_bar = st.progress(5, text="Preparing your document…")
            ocr_pages_used = 0
            scanned_pages_detected = 0
            ocr_pages_attempted = 0
            ocr_warning: str | None = None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = Path(tmp_file.name)

            progress_bar.progress(25, text=(
                "Extracting text from PDF…"
                if st.session_state.developer_mode
                else "Getting to know your document…"
            ))

            doc = fitz.open(str(tmp_path))
            extracted_text = ""
            page_docs = []
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text") or ""
                final_page_text = page_text
                if st.session_state.enable_ocr and _is_likely_scanned_page(page_text):
                    scanned_pages_detected += 1
                    ocr_pages_attempted += 1
                    ocr_text, ocr_error = _ocr_page_text(page)
                    if ocr_error and ocr_warning is None:
                        ocr_warning = ocr_error
                    if ocr_text:
                        final_page_text = ocr_text
                        ocr_pages_used += 1
                extracted_text += f"\n--- Page {page_num} ---\n"
                extracted_text += final_page_text + "\n"
                page_docs.append(Document(page_content=final_page_text, metadata={"page": page_num}))
            doc.close()
            progress_bar.progress(45, text=(
                "Text extracted. Preparing chunking…"
                if st.session_state.developer_mode
                else "Preparing your document for questions…"
            ))

            extracted_char_count = len(extracted_text.strip())
            if st.session_state.developer_mode:
                st.caption(
                    f"Extraction: **{len(page_docs)}** pages · **{extracted_char_count:,}** characters"
                )
            _render_ocr_status(
                enable_ocr=st.session_state.enable_ocr,
                scanned_pages_detected=scanned_pages_detected,
                ocr_pages_attempted=ocr_pages_attempted,
                ocr_pages_used=ocr_pages_used,
                ocr_warning=ocr_warning,
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
                else "Organizing content…"
            ):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    add_start_index=True,
                )
                docs_to_split = (
                    split_documents_by_legal_headers(page_docs)
                    if SPLIT_ON_LEGAL_HEADERS
                    else page_docs
                )
                chunks = text_splitter.split_documents(docs_to_split)
                page_texts = {
                    doc.metadata["page"]: doc.page_content
                    for doc in page_docs
                    if isinstance(doc.metadata.get("page"), int)
                }
                annotate_chunk_sections(chunks, page_texts)
            progress_bar.progress(65, text=(
                "Chunks created. Loading embedding model…"
                if st.session_state.developer_mode
                else "Almost ready…"
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
                    else "Almost ready for your first question…"
                ))
                with st.spinner(
                    f"Generating embeddings with {EMBEDDING_MODEL} & building FAISS index…"
                    if st.session_state.developer_mode
                    else "Making your document searchable…"
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
                    "Indexing complete." if st.session_state.developer_mode else "Good to go.",
                )

                if st.session_state.developer_mode:
                    _set_dev_index_logs(
                        faiss_msg=(
                            f"FAISS index {'re-' if had_existing_index else ''}created "
                            f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
                        ),
                    )
                else:
                    st.toast("Document indexed — you can chat.", icon="✅")

                _set_processed_document(file_name, uploaded_hash)
                _set_indexed_doc_stats(
                    file_name=file_name,
                    pages=len(page_docs),
                    chars=extracted_char_count,
                    reused_index=False,
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
for msg_idx, message in enumerate(st.session_state.messages):
    turn_label = f"turn {msg_idx // 2 + 1}"
    question_preview = message.get("for_question") if message["role"] == "assistant" else None
    if message["role"] == "user" and not question_preview:
        question_preview = _question_preview(message.get("content", ""))

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources_panel(
                message["sources"],
                developer_mode=st.session_state.developer_mode,
                title=_sources_panel_title(message),
            )
            if st.session_state.developer_mode:
                _render_source_checklist(
                    message["sources"],
                    target_section=message.get("target_section"),
                    title=_dev_panel_title(
                        "📋 Source checklist (eval)",
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
                            "📄 Exact Context fed to LLM",
                            message.get("for_question"),
                            fallback=turn_label,
                        ),
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

        def _thinking_timer_loop() -> None:
            if script_ctx is not None:
                add_script_run_ctx(threading.current_thread(), script_ctx)
            t0 = time.perf_counter()
            while not stop_timer.wait(0.12):
                elapsed = time.perf_counter() - t0
                timer_placeholder.caption(f"⏱ {elapsed:.1f}s · Thinking…")

        with st.spinner("Thinking…"):
            timer_thread = None
            if script_ctx is not None:
                timer_thread = threading.Thread(
                    target=_thinking_timer_loop,
                    daemon=True,
                    name="thinking-timer",
                )
                timer_thread.start()
            try:
                total_start = time.perf_counter()
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
                    context = _assemble_context(
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
                        try:
                            generation_start = time.perf_counter()
                            st.session_state.last_answer = generate_answer(
                                context=context,
                                query=query,
                                dummy_mode=st.session_state.dummy_generator_only,
                            )
                            generation_elapsed_ms = (
                                time.perf_counter() - generation_start
                            ) * 1000.0
                            st.session_state.last_generation_error = None
                        except Exception as e:
                            generation_error = e
                            generation_elapsed_ms = (
                                time.perf_counter() - generation_start
                            ) * 1000.0
                            st.session_state.last_answer = None
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

                total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            finally:
                if timer_thread is not None:
                    stop_timer.set()
                    timer_thread.join(timeout=5.0)
        timer_placeholder.empty()

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
                    _dev_panel_title("📊 Retrieval metrics (last run)", query_preview, fallback="live"),
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
                _dev_panel_title("📄 Exact Context fed to LLM", query_preview, fallback="live"),
                expanded=False,
            ):
                st.code(context, language="text")
                st.caption(f"• {len(raw_results)} chunks • {len(context)} chars • top-k={TOP_K}")
            with st.expander(
                _dev_panel_title("🔍 Retrieved raw chunks + scores", query_preview, fallback="live"),
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
        source_payload = _build_sources_payload(retrieved_docs, query=query)
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
