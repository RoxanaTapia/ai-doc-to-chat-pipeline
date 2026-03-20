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

Try the **[live demo](https://ai-doc-to-chat-demo.streamlit.app)** right now — drag a PDF, ask a question, and see cited answers.  
**Cut document review time by up to 40%** and turn your data into actionable insights with zero data exposure.

### Project Milestones (March 2026)

- ✅ Milestone 1 — Local prototype (upload + chat shell)
- ✅ Milestone 2 — Reliable PDF extraction (including OCR fallback)
- ✅ Milestone 3 — Chunking + embeddings + FAISS retrieval
- ✅ Milestone 4 — Basic RAG generation and answer display
- ✅ Milestone 5 — Live demo deployment completed ([ai-doc-to-chat-demo.streamlit.app](https://ai-doc-to-chat-demo.streamlit.app))
- ✅ Milestone 6 (`v0.6.0`) — First stable cloud-ready release

Last tagged release: `v0.6.0`.

### Run Locally From Scratch (Docker + Ollama)

1) Install Docker Desktop (Apple Silicon or Intel), launch it, and verify:

```bash
docker --version
docker run hello-world
```

2) Clone project and install Python dependencies:

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) Start Ollama server in Docker on port `11435`:

```bash
docker run -d --name ollama-cpu --restart unless-stopped -p 11435:11434 -v ollama:/root/.ollama ollama/ollama
# If container already exists:
# docker start ollama-cpu
```

4) Pull a model into that same Ollama server:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 ollama pull phi3:mini
# Optional larger model:
# OLLAMA_HOST=http://127.0.0.1:11435 ollama pull llama3.1:8b
```

5) Verify Ollama is reachable:

```bash
curl http://127.0.0.1:11435/api/tags
```

6) Run Streamlit against Docker Ollama:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 OLLAMA_MODEL=phi3:mini streamlit run src/app.py
```

Open `http://localhost:8501`, upload a PDF, and ask questions.

### Recommended Docker Resources

- CPU: `8-10` cores
- Memory: `16 GB` minimum (`24 GB` recommended for `llama3.1:8b`)
- Swap: `2-4 GB`

If generation is slow or fails with memory errors, use `phi3:mini` and/or lower context (`OLLAMA_NUM_CTX=2048`).

### Apple M5 / Metal Compatibility Note

On some M5/macOS combinations, native Ollama can crash in the Metal backend (`MTLLibraryErrorDomain`).  
Running Ollama inside Docker is the recommended workaround until upstream Metal fixes are fully stable across builds.

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
