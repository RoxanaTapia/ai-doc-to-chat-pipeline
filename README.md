# AI Doc-to-Chat Pipeline

Upload documents → Extract & Classify → Ask anything via intelligent RAG Chatbot

[Python](https://img.shields.io/badge/Python-3.12+-blue.svg) https://www.python.org/
[Streamlit](https://img.shields.io/badge/Streamlit-1.54+-green) https://streamlit.io/
[LangChain](https://img.shields.io/badge/LangChain-1.2+-orange) https://python.langchain.com/
[FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-9cf) https://github.com/facebookresearch/faiss
[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) https://opensource.org/licenses/MIT
[GitHub stars](https://img.shields.io/github/stars/RoxanaTapia/ai-doc-to-chat-pipeline?style=social) https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/stargazers

[Milestone 1](https://img.shields.io/badge/Milestone%201-Complete-brightgreen) https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline
[Milestone 2](https://img.shields.io/badge/Milestone%202-Complete-brightgreen) https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline
[Milestone 3](https://img.shields.io/badge/Milestone%203-In%20Progress-orange) https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline

Turn PDFs, scans, contracts, invoices or reports into a reliable, hallucination-free assistant — answers grounded in your actual documents.

Why this project?

Built with enterprise-grade principles (inspired by optimizations that cut document review time by 40%+ in production), this pipeline is:

- Ethical & private — no training on client data, local-first design
- Modular & configurable — YAML-driven, easy to extend
- Production-minded — logging, validation, graceful error handling
- Ideal for legal review, compliance, internal knowledge bases, invoice querying

In 2026, demand for private, controllable RAG document assistants remains high — especially on platforms like Upwork for sensitive/professional workflows.

Key Features

- Robust extraction: PyMuPDF + Tesseract OCR (digital + scanned PDFs)
- Document classification: Hugging Face zero-shot models
- Accurate RAG: LangChain + local FAISS vector store
- Clean UI: Streamlit — drag & drop + natural-language chat
- Fully configurable: chunk size, embeddings, prompts, LLM choice
- Security first: no data leaves your machine unless you choose otherwise
- Scalable: easy path to Pinecone, Chroma, Weaviate, etc.

Why This Over Public Tools (ChatGPT, Grok, Claude, Gemini)?

Public assistants are great for casual use — but professionals handling regulated/sensitive data need more.

Requirement                          | Public Cloud Tools                  | This Local RAG Pipeline                     | Winner for Pros
--------------------------------------|-------------------------------------|---------------------------------------------|-----------------
Data never leaves your machine       | Uploaded to 3rd-party servers       | 100% local after model download             | This project
Regulated / privileged data          | Often restricted or complex         | No external exposure — air-gappable         | This project
Full auditability                    | Limited visibility                  | Shows chunks + similarity scores            | This project
Custom retrieval tuning              | Very restricted                     | Fully configurable via YAML                 | This project
No per-query cost                    | Usage-based pricing                 | One-time hardware cost                      | This project
Offline capability                   | Requires internet                   | Works fully offline                         | This project

Bottom line: Use public tools for quick/public tasks. Use this for client contracts, compliance, due diligence, IP, healthcare — where trust, control, and zero exposure matter most.

Quick Start (≈ 2–3 minutes)

git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline

# Create isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies (modern hardware recommended)
pip install -r requirements.txt

# Optional: copy env template
cp .env.example .env               # edit if using external LLMs later

# Launch
streamlit run src/app.py

→ Open http://localhost:8501
→ Upload documents → start asking questions

Current Status (February 2026)

- Milestone 1 – Working local prototype (UI + upload + echo) → Complete
- Milestone 2 – Document extraction & text preview → Complete
- Milestone 3 – Chunking + Embeddings + FAISS Indexing (in progress)
- Milestones 4–6 – RAG generation, live demo, tagged release (v0.1.0) → Pending

Planned Project Structure

ai-doc-to-chat-pipeline/
├── src/                  # Core code
│   ├── app.py            # Streamlit UI
│   ├── extraction.py
│   ├── classification.py
│   ├── rag.py
│   └── utils/            # config, logging, prompts
├── configs/              # YAML settings
├── data/                 # Sample docs (gitignored)
├── tests/
├── docs/
├── .env.example
├── requirements.txt
└── README.md

Contributing & Commercial Use

MIT licensed — fork, modify, use commercially.

Need a customized version?
I help clients add:

- Multi-user auth & RBAC
- Enterprise vector DB integration
- Domain-specific prompts & classification
- Persistent history & audit trails
- Cloud-native / Kubernetes deployment

Happy to discuss your needs — let's build exactly what you require.

Made with ❤️ for trustworthy document AI.
Roxana Tapia · February 2026
