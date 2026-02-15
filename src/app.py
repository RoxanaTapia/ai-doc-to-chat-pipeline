import streamlit as st
import fitz  # PyMuPDF (pymupdf package)
import os
import tempfile
import yaml
import time
from pathlib import Path

# LangChain imports for Milestone 3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")

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
except (FileNotFoundError, OSError, yaml.YAMLError, KeyError, TypeError) as exc:
    st.error(f"Configuration error in {CONFIG_PATH}: {exc}")
    st.stop()

# Session state init
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "last_processed_name" not in st.session_state:
    st.session_state.last_processed_name = None

st.title("Upload → Extract → Chat! 🚀")
st.markdown("RAG-powered Document AI chatbot – coming soon!")

uploaded_file = st.file_uploader("Upload PDF or document", type=["pdf"])

if st.button("🗑️ Clear current document", type="primary"):
    st.session_state.vector_store = None
    st.session_state.chunks = None
    st.session_state.current_file = None
    st.session_state.last_processed_name = None
    st.success("Document cleared!")
    st.rerun()

extracted_text = ""

if uploaded_file is not None:
    st.success(f"Received file: {uploaded_file.name}")

    tmp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Extract text with PyMuPDF
        doc = fitz.open(tmp_path)
        extracted_text = ""
        for page_num, page in enumerate(doc, start=1):
            extracted_text += f"\n--- Page {page_num} ---\n"
            extracted_text += page.get_text("text") + "\n"
        doc.close()

        st.info("Text extracted successfully! Preview below.")
        st.text_area("Extracted Text (first 2000 characters)", extracted_text[:2000], height=300)

        # --- Chunking & Indexing (only if new file) ---
        is_new_file = (
            st.session_state.last_processed_name is None or
            st.session_state.last_processed_name != uploaded_file.name
        )

        if is_new_file:
            with st.spinner("Splitting text into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    add_start_index=True,
                )
                chunks = text_splitter.create_documents([extracted_text])

            st.success(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

            # Debug: show first few chunks
            with st.expander("First 3 chunks (debug view)", expanded=False):
                for i, chunk in enumerate(chunks[:3], 1):
                    preview = chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
                    st.markdown(f"**Chunk {i}** ({len(chunk.page_content)} chars, start index: {chunk.metadata.get('start_index', 'N/A')})")
                    st.text(preview)

            st.session_state.chunks = chunks

            # Embed & index
            had_existing_index = st.session_state.vector_store is not None
            with st.spinner(f"Generating embeddings with {EMBEDDING_MODEL} & building FAISS index..."):
                start = time.time()

                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

                vector_store = FAISS.from_documents(
                    documents=st.session_state.chunks,
                    embedding=embeddings
                )

                st.session_state.vector_store = vector_store
                took = time.time() - start

            st.success(
                f"FAISS index {'re-' if had_existing_index else ''}created "
                f"with {vector_store.index.ntotal} vectors • took {took:.1f} s"
            )

            # Remember this file
            st.session_state.last_processed_name = uploaded_file.name
            st.session_state.current_file = uploaded_file.name

            if len(chunks) <= 2:
                st.warning("Very little text found in document. Search might not work well.")

        else:
            st.info("Same document detected — skipping re-processing.")

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        # Always remove temp file, even if extraction/indexing fails.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Query interface
query = st.text_input(
    "Ask a question about the document:",
    disabled=st.session_state.vector_store is None
)

# Show current document
if st.session_state.current_file:
    st.caption(f"Currently indexed: **{st.session_state.current_file}**")

if query and st.session_state.vector_store:
    with st.spinner("Searching FAISS index..."):
        retrieved_docs = st.session_state.vector_store.similarity_search_with_score(query, k=TOP_K)

    st.subheader("Top Relevant Chunks (Milestone 3 debug – semantic retrieval)")
    st.caption("💡 **Similarity score**: higher value = more similar content. Typical good matches are between 0.3–0.9")

    for rank, (doc, score) in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:450] + "..." if len(doc.page_content) > 450 else doc.page_content
        st.markdown(f"**Rank {rank}** – Similarity: {score:.4f}")
        st.caption(f"Source starts at character {doc.metadata.get('start_index', 'N/A')}")
        st.text_area("Chunk content", preview, height=160, key=f"retrieved_{rank}")

    st.info("Milestone 3 complete: semantic search works. Next → Milestone 4 (prompt + LLM generation using these chunks)")

elif query:
    st.warning("Upload & process a document first to enable search.")
