# Project Milestones – AI Doc-to-Chat

Current date: February 15, 2026  
Goal: Private, local RAG chatbot for legal documents (e.g., 50-page contracts → "non-compete >2 years?").

## Milestone 1 – Working local prototype
**Status:** ✅ Complete  
**Goal:** Minimal but functional Streamlit app with quick-start flow from README.  
**Key achievements:**
- Folder structure (src/, configs/)
- requirements.txt with core deps
- Basic UI: title + file uploader + dummy messages + query echo
- No crashes on upload

## Milestone 2 – Basic document extraction & text preview
**Status:** ✅ Complete  
**Goal:** Real PDF processing + readable text preview.  
**Key achievements:**
- PyMuPDF text extraction (page by page)
- Scrollable preview (first 2000 chars)
- Error handling + temp file cleanup + loading spinner
- "Text extracted successfully!" feedback

## Milestone 3 – Chunking, local embeddings & semantic search
**Status:** ✅ Complete  
**Goal:** Turn extracted text into searchable vectors with FAISS.  
**Key achievements:**
- RecursiveCharacterTextSplitter + configurable chunk size/overlap
- Per-page Document metadata for chunk sourcing ("page ~X" or "multi" fallback)
- sentence-transformers/all-MiniLM-L6-v2 embeddings (cached via @st.cache_resource)
- FAISS index rebuild on new content (SHA-256 hash dedup)
- Multi-step progress bar + finalize_progress helper for UX polish
- Debug view: top-k chunks + similarity scores + page/source info
- Graceful handling of empty/scanned docs (early exit + OCR hint)

## Upcoming Milestones (planned)
- Milestone 4 – LLM generation (local Ollama + prompt engineering)
- Milestone 5 – Answer + source highlighting + citations
- Milestone 6 – OCR for scanned PDFs (pytesseract)
- Milestone 7 – Deployment (Docker / Streamlit Community Cloud)

Time estimate total: 6–8 weeks. Current: ~3 weeks in.
