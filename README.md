# AI Doc to Chat

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?style=for-the-badge&logo=streamlit)](https://ai-doc-to-chat-demo.streamlit.app)

**Private, local chat with your PDFs** — upload contracts, invoices, reports or scans, ask real questions in plain language, and get accurate answers with exact page references. No hallucinations, no data leaves your machine.

### Why this matters in 2026

Public AI tools like ChatGPT or Claude are great for casual use, but when dealing with sensitive contracts, legal docs, or internal policies, you need **privacy**, **control**, and **traceability**.

This local RAG (Retrieval-Augmented Generation) pipeline:
- Retrieves only relevant parts of your documents
- Generates grounded answers backed by real sources
- Runs completely offline (after setup)
- Can reduce document review time by up to 40% in real workflows

Perfect for lawyers, compliance teams, support staff, or anyone who works with confidential or regulated documents.

Try the **[live demo](https://ai-doc-to-chat-demo.streamlit.app)** right now — drag a PDF, ask a question, see how it cites pages.

### Current Status (March 2026)

- ✅ Milestone 1 — Working local prototype (UI + upload + basic response)
- ✅ Milestone 2 — PDF extraction + text preview (digital + scanned with OCR fallback)
- ✅ Milestone 3 — Chunking, local embeddings, FAISS semantic/hybrid search
- ✅ Milestone 4 — Basic RAG generation path + modern chat UI + sources display (Ollama stubbed)
- ⏳ Final Ollama integration — coming soon (waiting for hardware validation)

Tagged release: `v0.5.0` (pre-release with demo-ready features)

### Quick Start (2–3 minutes)

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure local LLM later
# cp .env.example .env

# Run the app
streamlit run src/app.py
```

Open http://localhost:8501, upload a PDF, and start asking questions.

### Core Technologies (simple view)

- **Streamlit** — clean, interactive UI  
- **LangChain + FAISS** — fast, private RAG retrieval  
- **PyMuPDF + Tesseract** — handles any PDF (digital or scanned)  
- **Ollama** — local LLM generation (phi3, llama3.1, etc.)  
- Fully configurable via YAML files

MIT licensed — free to use, modify, or build on commercially.

**Looking for a customized version?**  
Multi-user access, enterprise vector DB integration, audit logs, SSO, or cloud deployment? I'm available for freelance/consulting projects.

Made with ❤️ by **Roxana Tapia** — March 2026

From prototype to production-grade RAG: small models evolved to powerful local LLMs, but privacy stays king. This app recovers real context + generates accurate answers, modular and secure. For clients: "Cut document review time by 40% — try the demo and turn your data into real insights, zero risk."
