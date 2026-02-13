import streamlit as st
import fitz  # PyMuPDF (pymupdf package)
import os
import tempfile
import yaml
from pathlib import Path

# LangChain imports for Milestone 3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")

# Load configuration
CONFIG_PATH = Path("configs/config.yaml")
if not CONFIG_PATH.exists():
    st.error("Missing configs/config.yaml – create it with chunking & embeddings settings.")
    st.stop()

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]
EMBEDDING_MODEL = config["embeddings"]["model_name"]
TOP_K = config.get("rag", {}).get("top_k", 4)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

st.title("Upload → Extract → Chat! 🚀")
st.markdown("RAG-powered Document AI chatbot – coming soon!")

uploaded_file = st.file_uploader("Upload PDF or document", type=["pdf"])

extracted_text = ""

if uploaded_file is not None:
    st.success(f"Received file: {uploaded_file.name}")

    try:
        # Save uploaded file to temporary location (safe & clean)
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

        # Clean up temp file
        os.unlink(tmp_path)

        st.info("Text extracted successfully! Preview below.")
        st.text_area("Extracted Text (first 2000 characters)", extracted_text[:2000], height=300)
        # --- Chunking (Milestone 3) ---
        with st.spinner("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,  # helpful for later source referencing
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

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

# Future chat interface (placeholder)
query = st.text_input("Ask a question about the document:")
if query:
    if extracted_text:
        st.write("**Simple echo response (RAG coming soon):**")
        st.write(f"Query: {query}")
        st.write("Answer would be generated from extracted text...")
    else:
        st.warning("Upload and extract a document first!")
