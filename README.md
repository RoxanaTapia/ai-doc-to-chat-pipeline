# AI Doc-to-Chat Pipeline

**Upload documents → Extract & Classify → Ask anything via intelligent RAG Chatbot**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-9cf)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transform your PDFs, scans, contracts, invoices or reports into a reliable, hallucination-free assistant that answers precise questions — everything grounded in your actual documents.**

### Why this project?

Designed with enterprise-grade principles (inspired by real-world optimizations that reduced document processing & review time by 40%+ in production environments), this pipeline is:

- ethical (no training on client data)
- modular and fully configurable
- production-ready foundations (logging, error handling, graceful degradation)
- easy to extend or hand over to customers

**Ideal use cases:**
- Automating customer support with internal policies & FAQs
- Accelerating legal / compliance document review
- Building fast internal knowledge bases
- Natural-language querying of invoices, contracts, financial reports, technical manuals

**Market reality (early 2026):**  
RAG-based document chatbots and intelligent document processing continue to show strong demand on Upwork and similar platforms — frequently among the higher-paying AI automation categories.

### Key Features

- **Robust document extraction** — PyMuPDF + Tesseract OCR (handles both native digital PDFs and scanned documents)
- **Intelligent classification** — Hugging Face zero-shot / lightweight fine-tuned models to detect document type & key sections
- **Accurate Retrieval-Augmented Generation (RAG)** — LangChain + local FAISS vector store (fast, private, zero cloud cost at start)
- **Clean interactive UI** — Streamlit app: drag & drop documents → ask natural-language questions
- **100% configurable** — YAML files control chunking strategy, embedding model, LLM choice, prompt templates, etc.
- **Production-minded** — structured logging, input validation, graceful fallbacks, no hard crashes on malformed files
- **Scalability path** — trivial to switch to Pinecone, Chroma Cloud, Weaviate, Qdrant, etc.
- **Security & ethics first** — local-first design, never sends your documents to external providers unless you explicitly configure it

### Quick Start (≈ 2–3 minutes)

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline

# Create isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional but recommended) Provide API keys if using paid LLMs / embeddings
cp .env.example .env               # then edit .env

# Launch the application
streamlit run src/app.py
```
→ Open http://localhost:8501 in your browser  
→ Upload one or more documents → start asking questions

### Planned Project Structure

ai-doc-to-chat-pipeline/
├── src/                        # Código principal de la aplicación
│   ├── __init__.py             # (opcional, pero útil si algún día lo haces paquete)
│   ├── app.py                  # Entry point + UI Streamlit
│   ├── extraction.py           # o ocr_extractor.py (tu nombre es más explícito → prefiero)
│   ├── classification.py       # o classifier.py
│   ├── rag.py                  # o rag_core.py (tu nombre es más claro)
│   └── utils/                  # logging, config loader, helpers, prompts, etc.
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── configs/                    # o config/ — prefiero plural
│   ├── default.yaml
│   └── local.yaml              # (opcional para overrides locales)
├── data/                       # gitignored — PDFs de prueba, invoices de muestra, etc.
├── tests/
│   ├── test_extraction.py
│   ├── test_classification.py
│   └── test_rag.py
├── docs/                       # (opcional pero recomendado para proyectos serios)
│   └── architecture.md         # o usage.md, decisions.md, etc.
├── .github/
│   └── workflows/
│       └── ci.yml              # tests + lint en cada push/PR
├── .env.example
├── requirements.txt
├── README.md
└── LICENSE

### Contributing & Commercial Use

MIT licensed — feel free to fork, modify, and use this project in any personal or commercial context.

**Looking for a production-grade or customized version?**  
I regularly help clients adapt this kind of solution to their exact needs, including:

- Multi-user authentication & role-based access
- Integration with enterprise vector databases
- Fine-tuned classification or domain-specific prompts
- Persistent chat history & audit logging
- SSO / internal auth integration
- Kubernetes / cloud-native deployment patterns

Happy to discuss your requirements — let's build exactly what your organization needs.

Made with ❤️ for accurate, grounded, and trustworthy document AI automation.  
Roxana Tapia · February 2026
