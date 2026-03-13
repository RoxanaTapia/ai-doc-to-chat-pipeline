import streamlit as st
import fitz  # PyMuPDF (pymupdf package)
import tempfile
import yaml
import time
import hashlib
import os
from pathlib import Path

# LangChain imports for Milestone 3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag import generate_answer

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")
MAX_CHAT_MESSAGES = 40


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
APP_ROOT = Path(__file__).resolve().parent.parent
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
    TOP_K = config.get("rag", {}).get("top_k", 4)
    FETCH_K = config.get("rag", {}).get("fetch_k", 50)
    EARLY_PAGE_MAX = config.get("rag", {}).get("early_page_max", 20)
except (FileNotFoundError, OSError, yaml.YAMLError, KeyError, TypeError) as exc:
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


def _distance_to_ui_similarity(distance: float) -> float:
    """
    Convert vector distance to a bounded [0,1] similarity for UI display.
    Lower distance -> higher similarity.
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def _assign_similarity_scores_for_fallback(query: str, docs: list[Document]) -> None:
    """
    Ensure fallback-retrieved docs have numeric similarity for citation display.
    Uses embedding model query-doc pair scoring for docs lacking a score.
    """
    if not docs:
        return
    try:
        embeddings = get_embeddings(EMBEDDING_MODEL)
        scores = embeddings.embed_query(query)
        # Fallback similarity from model-space relation to keep score numeric.
        query_norm = sum(v * v for v in scores) ** 0.5
        doc_vecs = embeddings.embed_documents([doc.page_content for doc in docs])
        for doc, doc_vec in zip(docs, doc_vecs):
            doc_norm = sum(v * v for v in doc_vec) ** 0.5
            if query_norm == 0.0 or doc_norm == 0.0:
                doc.metadata["similarity"] = 0.0
                continue
            dot = sum(a * b for a, b in zip(scores, doc_vec))
            cosine = dot / (query_norm * doc_norm)
            doc.metadata["similarity"] = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    except Exception:
        for doc in docs:
            doc.metadata.setdefault("similarity", 0.0)


def _append_chat_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    """Append a chat message and prune old history."""
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)
    if len(st.session_state.messages) > MAX_CHAT_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_CHAT_MESSAGES:]


def _looks_like_temp_generator_fallback(answer: str) -> bool:
    """Detect the temporary placeholder returned by generate_answer fallback."""
    marker = "Temporary answer (Ollama still loading or running slowly)"
    return marker in (answer or "")


def _render_ollama_recovery_help(reason: str) -> None:
    """Show concise, actionable local recovery steps for generator issues."""
    st.warning(reason)
    with st.expander("How to fix local generation", expanded=False):
        st.markdown("Run these commands in a terminal:")
        st.code(
            "ollama serve\n"
            "ollama pull phi3:mini\n"
            "ollama list",
            language="bash",
        )
        st.caption("If `phi3:mini` appears in `ollama list`, ask again in chat.")


def _stream_text_chunks(text: str, chunk_size: int = 40):
    """Yield text in small chunks for progressive chat rendering."""
    safe_text = text or ""
    for i in range(0, len(safe_text), chunk_size):
        yield safe_text[i:i + chunk_size]


# Session state init
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "last_processed_name" not in st.session_state:
    st.session_state.last_processed_name = None
if "last_processed_hash" not in st.session_state:
    st.session_state.last_processed_hash = None
if "uploader_key_version" not in st.session_state:
    st.session_state.uploader_key_version = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []
if "last_retrieval_mode" not in st.session_state:
    st.session_state.last_retrieval_mode = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "developer_mode" not in st.session_state:
    st.session_state.developer_mode = _presentation_mode() == "developer"
if "allow_mode_toggle" not in st.session_state:
    # Keep toggle visibility stable during a session to avoid disappearing controls.
    st.session_state.allow_mode_toggle = (
        (not _is_probably_streamlit_cloud()) or st.session_state.developer_mode
    )

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

st.title("Upload → Extract → Chat! 🚀")
st.markdown("RAG-powered Document AI chatbot – coming soon!")

uploader_key = f"pdf_uploader_{st.session_state.uploader_key_version}"
uploaded_file = st.file_uploader("Upload PDF or document", type=["pdf"], key=uploader_key)

controls_col1, controls_col2 = st.columns(2)
if controls_col1.button("🗑️ Clear current document", type="primary"):
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.current_file = None
    st.session_state.last_processed_name = None
    st.session_state.last_processed_hash = None
    st.session_state.last_query = None
    st.session_state.last_retrieved_docs = []
    st.session_state.last_retrieval_mode = None
    st.session_state.last_answer = None
    st.session_state.messages = []
    st.session_state.uploader_key_version += 1
    st.success("Document cleared!")
    st.rerun()
if controls_col2.button("💬 Clear chat only"):
    st.session_state.last_query = None
    st.session_state.last_retrieved_docs = []
    st.session_state.last_retrieval_mode = None
    st.session_state.last_answer = None
    st.session_state.messages = []
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
            page_text = page.get_text("text")
            extracted_text += f"\n--- Page {page_num} ---\n"
            extracted_text += page_text + "\n"
            page_docs.append(Document(page_content=page_text, metadata={"page": page_num}))
        doc.close()
        progress_bar.progress(45, text="Text extracted. Preparing chunking...")

        extracted_char_count = len(extracted_text.strip())
        st.success("Extraction complete.")
        st.caption(
            f"Pages detected: {len(page_docs)} | Characters extracted: {extracted_char_count:,}"
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

            st.success(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
            if not chunks:
                st.session_state.vector_store = None
                st.session_state.chunks = []
                st.session_state.last_processed_name = uploaded_file.name
                st.session_state.last_processed_hash = uploaded_hash
                st.session_state.current_file = uploaded_file.name
                st.session_state.last_query = None
                st.session_state.last_retrieved_docs = []
                st.session_state.last_retrieval_mode = None
                st.session_state.last_answer = None
                st.session_state.messages = []
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

                st.success(
                    f"FAISS index {'re-' if had_existing_index else ''}created "
                    f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
                )

                st.session_state.last_processed_name = uploaded_file.name
                st.session_state.last_processed_hash = uploaded_hash
                st.session_state.current_file = uploaded_file.name
                st.session_state.last_query = None
                st.session_state.last_retrieved_docs = []
                st.session_state.last_retrieval_mode = None
                st.session_state.last_answer = None
                st.session_state.messages = []

                if len(chunks) <= 2:
                    st.warning("Very little text found in document. Search might not work well.")

        else:
            finalize_progress(progress_bar, "Same document detected. Reusing existing index.")
            st.session_state.current_file = uploaded_file.name
            st.info("Same document detected — skipping re-processing.")

    except Exception as e:
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

if query and st.session_state.vector_store:
    with st.chat_message("user"):
        st.markdown(query)
    _append_chat_message("user", query)

    with st.spinner("Searching FAISS index with early-page preference..."):
        # Retrieve candidates with scores, then apply page filter in Python for compatibility.
        candidate_results = st.session_state.vector_store.similarity_search_with_score(
            query,
            k=FETCH_K,
        )
        early_page_docs = []
        for doc, raw_distance in candidate_results:
            page = doc.metadata.get("page")
            doc.metadata["similarity"] = _distance_to_ui_similarity(float(raw_distance))
            if isinstance(page, int) and page <= EARLY_PAGE_MAX:
                early_page_docs.append(doc)

        if early_page_docs:
            retrieved_docs = early_page_docs[:TOP_K]
            st.session_state.last_retrieval_mode = "early_page_filtered"
        else:
            # Fallback to global MMR when no early-page candidates are available.
            retrieved_docs = st.session_state.vector_store.max_marginal_relevance_search(
                query,
                k=TOP_K,
                fetch_k=FETCH_K,
                lambda_mult=0.5,
            )
            st.session_state.last_retrieval_mode = "global_mmr_fallback"
            st.info(
                f"No matches found in pages <= {EARLY_PAGE_MAX}. "
                "Showing best matches from all pages."
            )
            _assign_similarity_scores_for_fallback(query, retrieved_docs)

    st.session_state.last_query = query
    st.session_state.last_retrieved_docs = retrieved_docs
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    try:
        with st.spinner("Generating answer with local Ollama model..."):
            st.session_state.last_answer = generate_answer(query=query, context=context)
            if _looks_like_temp_generator_fallback(st.session_state.last_answer):
                _render_ollama_recovery_help(
                    "Using temporary fallback answer because local Ollama generation is not ready yet."
                )
    except Exception as e:
        st.session_state.last_answer = None
        error_text = str(e).lower()
        if "connection" in error_text or "refused" in error_text:
            reason = "Could not connect to Ollama. Start the Ollama server first."
        elif "not found" in error_text or "model" in error_text:
            reason = "The model `phi3:mini` is unavailable locally. Pull it, then retry."
        elif "timeout" in error_text or "timed out" in error_text:
            reason = "Local model timed out. Retry with a shorter question or smaller context."
        else:
            reason = "Could not generate an Ollama answer right now."
        _render_ollama_recovery_help(reason)
        if st.session_state.developer_mode:
            st.caption(f"Generator error details: {e}")

    assistant_message = (
        st.session_state.last_answer
        or "I could not generate an answer at this time. Please try again."
    )
    source_payload = _build_sources_payload(retrieved_docs)
    with st.chat_message("assistant"):
        rendered_answer = st.write_stream(_stream_text_chunks(assistant_message))
        if source_payload:
            with st.expander("Sources", expanded=False):
                for source_idx, source in enumerate(source_payload, start=1):
                    st.markdown(
                        f"**Source {source_idx}** (similarity score: {source['score']}) - page {source['page']}"
                    )
                    st.code(source["preview"], language="text")
                    st.markdown("---")
    final_assistant_text = (
        rendered_answer if isinstance(rendered_answer, str) else assistant_message
    )
    _append_chat_message("assistant", final_assistant_text, sources=source_payload)

if st.session_state.last_retrieved_docs and st.session_state.developer_mode:
    st.subheader("Top Relevant Chunks (Developer debug view)")
    if st.session_state.last_retrieval_mode == "early_page_filtered":
        st.caption(
            f"💡 **Early-page filtered retrieval active** (pages <= {EARLY_PAGE_MAX}). "
            "Use ranking order as the primary signal."
        )
    else:
        st.caption(
            "💡 **MMR reranking active** (diversity + relevance) on all pages. "
            "Use ranking order as the primary signal."
        )

    for rank, doc in enumerate(st.session_state.last_retrieved_docs, 1):
        preview = doc.page_content[:450] + "..." if len(doc.page_content) > 450 else doc.page_content
        page_label = doc.metadata.get("page", "N/A") if isinstance(doc.metadata.get("page"), int) else "multi"
        st.markdown(f"**Rank {rank}**")
        st.caption(
            f"Source page ~{page_label} • "
            f"starts at character {doc.metadata.get('start_index', 'N/A')}"
        )
        st.text_area("Chunk content", preview, height=160, key=f"retrieved_{rank}")
