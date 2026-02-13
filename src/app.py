import streamlit as st
import fitz  # PyMuPDF (pymupdf package)
import os
import tempfile

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")

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
