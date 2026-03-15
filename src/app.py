import streamlit as st
import fitz  # PyMuPDF (pymupdf package)
import tempfile
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag import generate_answer, load_generation_config

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")
APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=APP_ROOT / ".env")
MAX_CHAT_MESSAGES = 40
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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
    """
    override = (os.getenv("APP_PRESENTATION_MODE") or "").strip().lower()
    if override in {"client", "developer"}:
        return override
    return "client" if _is_probably_streamlit_cloud() else "developer"


if _is_probably_streamlit_cloud():
    st.sidebar.info(
        "This is a **public demo** hosted on Streamlit Cloud — documents are temporarily uploaded here. "
        "For sensitive files, run the app **locally** (clone the repo & streamlit run src/app.py)."
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


def _build_sources_payload(retrieved_docs: list[Document]) -> list[dict]:
    """Create a compact serializable source payload per assistant answer."""
    payload = []
    for chunk in retrieved_docs:
        preview = chunk.page_content[:100]
        similarity = chunk.metadata.get("similarity")
        payload.append(
            {
                "page": chunk.metadata.get("page", "N/A"),
                "score": round(similarity, 3) if isinstance(similarity, (float, int)) else "N/A",
                "preview": preview,
            }
        )
    return payload


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


def _append_chat_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    """Append a chat message and prune old history."""
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > MAX_CHAT_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_MESSAGES:]


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
    """Apply optional reranker and return final top-k retrieval results."""
    if not st.session_state.enable_reranker:
        return candidate_pool[:TOP_K], mode, None, 0.0

    rerank_start = time.perf_counter()
    raw_results, reranker_warning = _apply_reranker(
        query,
        candidate_pool,
        top_n=RERANKER_TOP_N,
        model_name=RERANKER_MODEL_NAME,
    )
    rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000.0
    final_mode = f"{mode}_reranked"
    return raw_results, final_mode, reranker_warning, rerank_elapsed_ms


def _render_ollama_recovery_help(reason: str) -> None:
    """Show concise, actionable local recovery steps for generator issues."""
    try:
        model_name = load_generation_config().get("model", "phi3:mini")
    except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
        model_name = "phi3:mini"
    preferred_models = [model_name, "phi3.5:mini", "phi3:mini"]
    # Keep order stable while removing duplicates.
    recommended_models = list(dict.fromkeys(preferred_models))
    pull_commands = "\n".join(f"ollama pull {name}" for name in recommended_models)

    st.warning(reason)
    with st.expander("How to fix local generation", expanded=False):
        st.markdown("Run these commands in a terminal:")
        st.code(
            "ollama serve\n"
            f"{pull_commands}\n"
            "ollama list",
            language="bash",
        )
        recommended_order_text = " -> ".join(f"`{name}`" for name in recommended_models)
        st.caption(
            "Preferred order for CPU smoke tests: "
            f"{recommended_order_text}."
        )


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
    st.session_state.messages = []


def _reset_document_state(*, bump_uploader_key: bool = False) -> None:
    """Reset the indexed document and all retrieval/generation state."""
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.current_file = None
    st.session_state.last_processed_name = None
    st.session_state.last_processed_hash = None
    st.session_state.bm25_state = None
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
        "use_page_separators": False,
        "dummy_generator_only": _env_bool("USE_DUMMY_GENERATOR", True),
        "bm25_state": None,
        "retrieval_strategy": (
            RETRIEVAL_STRATEGY_DEFAULT
            if RETRIEVAL_STRATEGY_DEFAULT in {"semantic", "hybrid"}
            else "semantic"
        ),
        "enable_reranker": RERANKER_ENABLED_DEFAULT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "allow_mode_toggle" not in st.session_state:
        # Keep toggle visibility stable during a session to avoid disappearing controls.
        st.session_state.allow_mode_toggle = (
            (not _is_probably_streamlit_cloud()) or st.session_state.developer_mode
        )


_init_session_state()

if st.session_state.allow_mode_toggle:
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
        Upload PDFs -> Extract -> Chat! 🚀

        A local, private RAG chatbot for contracts, scans, and reports.
        Built with LangChain + FAISS + Streamlit.
        """
    )
st.sidebar.markdown("**Tech stack**")
tech_tags = ["Python", "RAG", "LangChain", "FAISS", "Streamlit", "OCR", "PDF", "Local LLM"]
st.sidebar.caption(" | ".join(tech_tags))

st.session_state.enable_ocr = st.sidebar.toggle(
    "Enable OCR for scanned pages",
    value=st.session_state.enable_ocr,
    help="Attempts OCR only on pages with little/no extractable text.",
)
st.session_state.use_page_separators = st.sidebar.checkbox(
    "Use page separators in context",
    value=st.session_state.use_page_separators,
    help="Adds page labels between chunks when assembling context for generation.",
)
st.session_state.dummy_generator_only = st.sidebar.checkbox(
    "Use dummy generator only (for testing)",
    value=st.session_state.dummy_generator_only,
    help="ON: echo mode answer. OFF: try local Ollama and fallback if unavailable.",
)
if st.session_state.developer_mode:
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

st.title("Upload → Extract → Chat! 🚀")
st.markdown("RAG-powered Document AI chatbot – coming soon!")

uploader_key = f"pdf_uploader_{st.session_state.uploader_key_version}"
uploaded_file = st.file_uploader("Upload PDF or document", type=["pdf"], key=uploader_key)

controls_col1, controls_col2 = st.columns(2)
if controls_col1.button("🗑️ Clear current document", type="primary"):
    _reset_document_state(bump_uploader_key=True)
    st.success("Document cleared!")
    st.rerun()
has_chat_history = bool(st.session_state.messages)
if controls_col2.button(
    "💬 Clear chat only",
    disabled=not has_chat_history,
    help="No chat history yet." if not has_chat_history else None,
):
    _reset_chat_state()
    st.success("Chat cleared. Indexed document remains available.")
    st.rerun()

extracted_text = ""

if uploaded_file is not None:
    st.success(f"Received file: {uploaded_file.name}")

    tmp_path = None
    progress_bar = None
    try:
        progress_bar = st.progress(5, text="Preparing uploaded file...")
        file_bytes = uploaded_file.getvalue()
        uploaded_hash = hashlib.sha256(file_bytes).hexdigest()
        ocr_pages_used = 0
        scanned_pages_detected = 0
        ocr_pages_attempted = 0
        ocr_warning: str | None = None

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        progress_bar.progress(25, text="Extracting text from PDF...")

        # Extract text with PyMuPDF
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
        progress_bar.progress(45, text="Text extracted. Preparing chunking...")

        extracted_char_count = len(extracted_text.strip())
        st.success("Extraction complete.")
        st.caption(
            f"Pages detected: {len(page_docs)} | Characters extracted: {extracted_char_count:,}"
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

        # --- Chunking & Indexing (only if new file) ---
        is_new_file = st.session_state.last_processed_hash != uploaded_hash

        if is_new_file:
            with st.spinner("Splitting text into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.split_documents(page_docs)
            progress_bar.progress(65, text="Chunks created. Loading embedding model...")

            if st.session_state.developer_mode:
                st.success(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
            if not chunks:
                _reset_document_state()
                st.session_state.chunks = []
                _set_processed_document(uploaded_file.name, uploaded_hash)
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
                progress_bar.progress(80, text="Embeddings ready. Building FAISS index...")
                with st.spinner(f"Generating embeddings with {EMBEDDING_MODEL} & building FAISS index..."):
                    start = time.time()

                    embeddings = get_embeddings(EMBEDDING_MODEL)

                    vector_store = FAISS.from_documents(
                        documents=st.session_state.chunks,
                        embedding=embeddings
                    )

                    st.session_state.vector_store = vector_store
                    took = time.time() - start
                finalize_progress(progress_bar, "Indexing complete.")

                if st.session_state.developer_mode:
                    st.success(
                        f"FAISS index {'re-' if had_existing_index else ''}created "
                        f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
                    )
                else:
                    st.success("Document processed. You can now ask questions.")

                _set_processed_document(uploaded_file.name, uploaded_hash)

                if len(chunks) <= 2:
                    st.warning("Very little text found in document. Search might not work well.")

        else:
            finalize_progress(progress_bar, "Same document detected. Reusing existing index.")
            st.session_state.current_file = uploaded_file.name
            st.info("Same document detected — skipping re-processing.")

    except (OSError, ValueError, RuntimeError) as e:
        finalize_progress(progress_bar, "Processing failed.")
        st.error(f"Error processing PDF: {str(e)}")
        st.exception(e)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

# Chat history UI (persists across reruns)
assistant_turn_count = 0
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            assistant_turn_count += 1
            turn_number = assistant_turn_count
            with st.expander(f"Sources (turn {turn_number})", expanded=False):
                for source_idx, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {source_idx}** (similarity score: {source['score']}) - page {source['page']}"
                    )
                    st.code(source["preview"], language="text")
                    st.markdown("---")

if st.session_state.current_file:
    st.caption(f"Currently indexed: **{st.session_state.current_file}**")
elif st.session_state.vector_store is None:
    st.caption("Upload and process a document to enable chat.")

query = st.chat_input(
    "Ask a question about the document...",
    disabled=st.session_state.vector_store is None,
)

if query and query.strip() and st.session_state.vector_store:
    query = query.strip()
    with st.chat_message("user"):
        st.markdown(query)
    _append_chat_message("user", query)

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
    with st.spinner("Thinking..."):
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        try:
            candidate_limit = (
                max(TOP_K, RERANKER_TOP_N) if st.session_state.enable_reranker else TOP_K
            )
            candidate_pool, mode, first_stage_warning, retrieval_diag = _run_first_stage_retrieval(
                query,
                candidate_limit=candidate_limit,
            )
            raw_results, final_mode, reranker_warning, rerank_elapsed_ms = _finalize_retrieval_candidates(
                query,
                candidate_pool=candidate_pool,
                mode=mode,
            )
            st.session_state.last_retrieval_mode = final_mode
            retrieval_warning = reranker_warning or first_stage_warning
            retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0

            for doc, score in raw_results:
                doc.metadata["similarity"] = score
            retrieved_docs = [doc for doc, _score in raw_results]

            st.session_state.last_query = query
            st.session_state.last_raw_results = raw_results
            st.session_state.last_retrieved_docs = retrieved_docs
            context = _assemble_context(
                raw_results,
                use_page_separators=st.session_state.use_page_separators,
            )
            st.session_state.last_context = context
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
            retrieval_error = e
            retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0
            st.session_state.last_retrieval_mode = "retrieval_failed"
            raw_results = []
            retrieved_docs = []
            context = ""
            st.session_state.last_query = query
            st.session_state.last_raw_results = []
            st.session_state.last_retrieved_docs = []
            st.session_state.last_context = ""
            st.session_state.last_answer = None

        if retrieval_error is None:
            try:
                generation_start = time.perf_counter()
                st.session_state.last_answer = generate_answer(
                    context=context,
                    query=query,
                    dummy_mode=st.session_state.dummy_generator_only,
                )
                generation_elapsed_ms = (time.perf_counter() - generation_start) * 1000.0
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                generation_error = e
                generation_elapsed_ms = (time.perf_counter() - generation_start) * 1000.0
                st.session_state.last_answer = None
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
        if st.session_state.last_retrieval_metrics:
            with st.expander("📊 Retrieval metrics (last run)", expanded=False):
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
        with st.expander("📄 Exact Context fed to LLM", expanded=False):
            st.code(context, language="text")
            st.caption(f"• {len(raw_results)} chunks • {len(context)} chars • top-k={TOP_K}")
        with st.expander("🔍 Retrieved raw chunks + scores", expanded=False):
            for i, (doc, _raw_score) in enumerate(raw_results, start=1):
                similarity = doc.metadata.get("similarity")
                st.markdown(
                    f"**Chunk {i}** (similarity score: {round(similarity, 3) if isinstance(similarity, (float, int)) else 'N/A'})  \n"
                    f"Page: {doc.metadata.get('page', '?')}"
                )
                st.code(doc.page_content, language="text")

    if retrieval_error is not None:
        retrieval_reason = (
            "I couldn't retrieve relevant document chunks right now. "
            "Please try again."
        )
        st.warning(retrieval_reason)
        if st.session_state.developer_mode:
            st.caption(f"Retrieval error details: {retrieval_error}")
        with st.chat_message("assistant"):
            st.markdown(retrieval_reason)
        _append_chat_message("assistant", retrieval_reason)
    elif generation_error is not None:
        e = generation_error
        st.session_state.last_answer = None
        try:
            model_name = load_generation_config().get("model", "phi3:mini")
        except (FileNotFoundError, OSError, yaml.YAMLError, TypeError, ValueError, KeyError):
            model_name = "phi3:mini"
        error_text = str(e).lower()
        if "connection" in error_text or "refused" in error_text:
            reason = "Could not connect to Ollama. Start the Ollama server first."
        elif "not found" in error_text or "model" in error_text:
            reason = f"The model `{model_name}` is unavailable locally. Pull it, then retry."
        elif "timeout" in error_text or "timed out" in error_text:
            reason = "Local model timed out. Retry with a shorter question or smaller context."
        else:
            reason = "Could not generate an Ollama answer right now."
        _render_ollama_recovery_help(reason)
        if st.session_state.developer_mode:
            st.caption(f"Generator error details: {e}")
        assistant_message = (
            "I couldn't generate an answer right now. "
            f"{reason}"
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
        _append_chat_message("assistant", assistant_message)
    else:
        assistant_message = (
            st.session_state.last_answer
            or "I could not generate an answer at this time. Please try again."
        )
        source_payload = _build_sources_payload(retrieved_docs)
        with st.chat_message("assistant"):
            st.write_stream(_stream_text_chunks(assistant_message))
            if source_payload:
                with st.expander("Sources", expanded=False):
                    for source_idx, source in enumerate(source_payload, start=1):
                        st.markdown(
                            f"**Source {source_idx}** (similarity score: {source['score']}) - page {source['page']}"
                        )
                        st.code(source["preview"], language="text")
                        st.markdown("---")
        _append_chat_message("assistant", assistant_message, sources=source_payload)
elif query is not None and st.session_state.vector_store:
    st.warning("Please enter a non-empty question.")
