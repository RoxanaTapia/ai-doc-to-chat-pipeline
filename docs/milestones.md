# Project Milestones – AI Doc-to-Chat

Current date: March 14, 2026  
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

## Milestone 4 – LLM generation (local Ollama + prompt engineering)
**Status:** ✅ Complete  
**Goal:** Generate grounded answers from retrieved context using a local model.  
**Key achievements:**
- Added `src/rag.py` generation module with YAML-driven prompt loading
- Local Ollama integration via `langchain-ollama` (`ChatOllama`)
- Configurable generation settings (`model`, `max_new_tokens`, `temperature`, `top_p`, `do_sample`, `num_ctx`, timeout)
- App-level recovery guidance when Ollama is unavailable/misconfigured
- Optional dummy mode and environment-variable overrides for local testing

## Milestone 5 – Answer + source citations
**Status:** ✅ Complete  
**Goal:** Show concise answers with verifiable source context in chat UX.  
**Key achievements:**
- Chat-style interface with persistent session history (`st.session_state.messages`)
- Per-answer source payload with page, similarity score, and chunk preview
- Collapsible **Sources** panel on each assistant turn
- Clear-chat behavior separate from clear-document behavior
- New document upload resets chat history; same document preserves chat history

## Milestone 6 – OCR fallback for scanned PDFs
**Status:** ✅ Complete  
**Goal:** Recover text from scanned pages when native PDF extraction is weak.  
**Key achievements:**
- Conditional OCR pass on likely scanned pages (short-text heuristic)
- OCR pipeline via `pytesseract` + PIL page rendering/preprocessing
- User-facing OCR diagnostics and warnings in the app
- Sidebar toggle to enable/disable OCR fallback at runtime
- Graceful handling when OCR dependencies/runtime are missing

## Milestone 4.7 – Production retrieval hardening (reranker / hybrid)
**Status:** 🚧 Implemented, pending hardware validation  
**Goal:** Improve long-document precision before generation.  
**Key achievements:**
- Added configurable retrieval strategy (`semantic` or `hybrid`) in `configs/config.yaml`
- Implemented hybrid retrieval (dense FAISS + BM25) with Reciprocal Rank Fusion (RRF)
- Added optional cross-encoder reranker stage for top candidates
- Added runtime developer controls for strategy switching and reranker toggling
- Added graceful fallback when BM25 or reranker dependencies are unavailable

## Upcoming Milestones (planned)
- Milestone 7 – Deployment (Docker / Streamlit Community Cloud)

Time estimate total: 6–8 weeks. Current: Milestones 1–6 complete.
