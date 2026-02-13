import streamlit as st

st.set_page_config(page_title="AI Doc-to-Chat", layout="wide")

st.title("Upload → Extract → Chat! 🚀")
st.markdown("RAG-powered Document AI chatbot – coming soon!")

uploaded_file = st.file_uploader("Upload PDF or document", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Received file: {uploaded_file.name}")
    st.info("Extraction & RAG pipeline placeholder – processing would start here.")

    # Future chat interface
    query = st.text_input("Ask a question about the document:")
    if query:
        st.write(f"Echo: {query}")
