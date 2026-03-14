# AI Doc-to-Chat Pipeline

**Upload documents → Extract & Classify → Ask anything via intelligent RAG Chatbot**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-green)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-9cf)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/RoxanaTapia/ai-doc-to-chat-pipeline?style=social)](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/RoxanaTapia/ai-doc-to-chat-pipeline)](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/commits/main)

[![Milestone 1](https://img.shields.io/badge/Milestone%201-Complete-brightgreen)](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline)
[![Milestone 2](https://img.shields.io/badge/Milestone%202-Complete-brightgreen)](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline)
[![Milestone 3](https://img.shields.io/badge/Milestone%203-In%20Progress-orange)](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline)

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

### Current Status (February 2026)

- ✅ **Milestone 1** – Working local prototype (UI + upload + echo)
- ✅ **Milestone 2** – Basic document extraction & text preview (PyMuPDF, spinner, error handling, page preview)
- 🚧 **Milestone 3** – Chunking + Embeddings + FAISS Indexing (in progress on branch `feat/milestone-3-chunking-embeddings-faiss`)
- ⏳ Milestones 4–6 – RAG generation, live demo, tagged release

Extraction and basic UI are functional. Semantic search (RAG retrieval) is next.

### Why Choose This Over Public Tools? (ChatGPT, Grok, Claude, Gemini, etc.)

In 2026, many excellent cloud-based AI assistants already let you upload PDFs and ask questions. So why build (or use) a fully local RAG pipeline like this one?

The answer lies in **privacy**, **control**, **cost**, and **compliance** — requirements that matter most to professionals handling sensitive or regulated documents.

| Requirement                              | Public Cloud Tools (ChatGPT / Grok / Claude / ...) | This Local RAG Pipeline                              | Typical Winner for Enterprise / Professional Use |
|------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|--------------------------------------------------|
| **Data never leaves your machine**       | Data uploaded to 3rd-party servers (even with "zero-retention" plans) | 100% offline & local after model download             | This project                                   |
| **Regulated / privileged data** (legal, finance, healthcare, M&A, government) | Often prohibited or requires complex enterprise agreements | No external exposure — air-gappable if needed         | This project                                   |
| **Full auditability & traceability**     | Limited visibility into retrieved chunks & scoring  | Shows exact retrieved chunks + similarity scores      | This project                                   |
| **Custom retrieval tuning** (chunk size, overlap, embedding model, re-ranking) | Very restricted or impossible                       | Fully configurable via `configs/config.yaml`          | This project                                   |
| **No recurring per-query / per-token cost** | Usage-based pricing scales quickly with volume     | One-time hardware cost (or existing laptop)           | This project (high-volume users)               |
| **Offline / no-internet scenarios**      | Requires constant connection                        | Works completely offline (after first model download) | This project                                   |
| **No file-size / page-count hard limits**| Provider-imposed caps (practical ~50–200 pages)    | Limited only by local hardware                        | This project                                   |
| **Explainable retrieval** (for lawyers / auditors) | Black-box context window                           | Full control over what context is sent to the model   | This project                                   |
| **Casual / one-off / non-sensitive use** | Extremely fast to start                             | Requires local setup & install                        | Public tools                                   |

**Bottom line for clients**

- If you're doing casual research, summarizing public reports, or working with non-sensitive internal notes → public tools are faster and simpler to start.
- If you're dealing with **client contracts**, **privileged communications**, **compliance reviews**, **due diligence**, **IP**, **patient data**, **merger documents**, or **anything regulated** → this local-first pipeline delivers **provable confidentiality**, **predictable cost**, **full customization**, and **audit-ready transparency** that no public cloud service can match without significant legal, contractual, and cost overhead.

This project is built exactly for those higher-stakes professional workflows — the ones where **trust**, **control**, and **zero external exposure** are non-negotiable.

Happy to adapt it further for your exact industry or compliance needs.

### Quick Start (≈ 2–3 minutes)

First-run checklist:

- [ ] Create and activate virtual environment (`python -m venv .venv` + `source .venv/bin/activate`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Start Ollama (`ollama serve`)
- [ ] Pull local model (`ollama pull phi3:mini`)
- [ ] Run app (`streamlit run src/app.py`)

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline

# Create isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
# Recommended (modern hardware — Apple Silicon, recent macOS, Linux, Windows):
pip install -r requirements.txt

# For older Intel Macs (e.g. Mid-2015 MacBook Pro on Monterey) or install issues:
# pip install -r requirements-legacy.txt

# (Optional) Configure cloud providers with a safe template (empty values by default)
cp .env.example .env
# then edit `.env` only on your machine

# Required for local answer generation (Milestone 4 skeleton):
# 1) Install Ollama: https://ollama.com/download
# 2) Start Ollama server (keep this terminal running):
ollama serve
# 3) Download the model used by this app:
ollama pull phi3:mini
# 4) (Optional) Quick model check in terminal:
ollama run phi3:mini

# Launch the application
streamlit run src/app.py
```
→ Open http://localhost:8501 in your browser  
→ Upload one or more documents → start asking questions

#### Presentation modes (client vs developer)

The app supports two UI modes:

- **Client mode** (clean demo): shows generated answer + source citations only.
- **Developer mode** (local QA): also shows retrieval debug panels (raw extraction preview + ranked chunks).

Default behavior:

- **Local run**: starts in **Developer mode**.
- **Streamlit Cloud** (when detected): starts in **Client mode**.

Optional explicit override:

```bash
# Force client presentation (recommended for live demos)
APP_PRESENTATION_MODE=client streamlit run src/app.py

# Force developer presentation (retrieval debug visible)
APP_PRESENTATION_MODE=developer streamlit run src/app.py
```

Deploy recommendation:

- In your deployed environment, set `APP_PRESENTATION_MODE=client` to guarantee the clean client-facing UI.
- Keep local development in `developer` mode when tuning retrieval quality.

#### Local LLM setup (Ollama) — required to see generated answers

This app currently calls a local Ollama model in `src/rag.py`:

- Expected model: `phi3:mini`
- If Ollama is not running (or the model is missing), retrieval still works but generation shows a recovery warning with commands to fix local setup.
- Keep `ollama serve` running while using Streamlit.
- `ollama run phi3:mini` is optional and only for manual terminal testing.

Quick verification commands:

```bash
ollama list               # confirms phi3:mini is downloaded
ollama run phi3:mini      # optional interactive check
```

If you want to use a different model, set `rag.generation.model` in `configs/config.yaml`
or export `OLLAMA_MODEL` in your environment.

Generation defaults are now configurable in `configs/config.yaml` under:

- `rag.generation.model`
- `rag.generation.max_new_tokens`
- `rag.generation.temperature`
- `rag.generation.top_p`
- `rag.generation.do_sample`
- `rag.generation.fallback_to_dummy_on_error`
- `rag.generation.num_ctx`
- `rag.generation.timeout_seconds`

Optional environment overrides (local `.env`) are supported:

- `USE_DUMMY_GENERATOR=true|false` (UI default for dummy mode checkbox)
- `OLLAMA_MODEL`
- `OLLAMA_MAX_NEW_TOKENS`
- `OLLAMA_TEMPERATURE`
- `OLLAMA_TOP_P`
- `OLLAMA_DO_SAMPLE`
- `OLLAMA_FALLBACK_TO_DUMMY`
- `OLLAMA_NUM_CTX`
- `OLLAMA_TIMEOUT_SECONDS`

#### Next-week Ollama smoke-test plan (hardware arrival)

Recommended first model:

- Start with `phi3.5:mini` if available on your machine (`ollama pull phi3.5:mini`).
- Keep `phi3:mini` as the fallback baseline if you want conservative compatibility.
- Alternative for slightly stronger quality (heavier): `llama3.1:8b` in quantized variants.

Suggested execution steps:

1. Keep `USE_DUMMY_GENERATOR=true` until runtime is ready.
2. Start Ollama and pull the chosen model.
3. Switch to real generation (`USE_DUMMY_GENERATOR=false` or uncheck the sidebar toggle).
4. Run 3-5 realistic questions on 1-2 contract PDFs.
5. Record latency, memory pressure, and grounding quality.

Suggested acceptance targets:

- Context budget: 12k-16k characters for retrieval context.
- Latency target: under ~8-12 seconds per answer on average CPU laptop.
- Stability target: no OOM on a 50-page contract with top-k retrieval.
- Quality target: grounded answers with correct page/chunk citations for at least 3 smoke questions.

#### OCR fallback for scanned PDFs

The app can automatically attempt OCR on pages with little/no native text:

- Sidebar toggle: **Enable OCR for scanned pages** (ON by default).
- OCR runs only on likely scanned pages (conditional fallback, not all pages).
- After extraction, the UI shows OCR diagnostics:
  - scanned pages detected
  - OCR applied
  - OCR unresolved

If OCR does not run locally, verify:

```bash
# macOS
brew install tesseract

# Python deps in your active venv
pip install -r requirements.txt
```

Notes:

- `pytesseract` is a Python wrapper; it still needs the system `tesseract` binary.
- OCR quality depends on scan quality; low-resolution pages may remain partially unresolved.

#### macOS Monterey / Intel Mac notes (Mid-2015 MacBook Pro or similar)
- Use Python 3.12 (via `brew install python@3.12` or python.org installer).
- If modern dependencies fail to install, use the legacy file:  
  `pip install -r requirements-legacy.txt` (pins Torch <2.3.0, older LangChain, etc.)
- Tesseract OCR: `brew install tesseract` (required for scanned PDFs).
- Performance is good for small/medium documents; for very large PDFs or heavy use, consider upgrading hardware or macOS.

### Planned Project Structure

```text
ai-doc-to-chat-pipeline/
├── src/                        # Core application code (modular & testable)
│   ├── init.py             # Makes src a package (optional but useful)
│   ├── app.py                  # Streamlit entry point & UI logic
│   ├── extraction.py           # PDF/text extraction (PyMuPDF + Tesseract OCR)
│   ├── classification.py       # Document/section type classification (Hugging Face)
│   ├── rag.py                  # RAG pipeline: embedding, retrieval, generation (LangChain + FAISS)
│   └── utils/                  # Shared helpers
│       ├── init.py
│       ├── config.py           # YAML config loading & validation
│       ├── logging.py          # Structured logging setup
│       └── prompts.py          # Prompt templates (easy to version & override)
├── configs/                    # Configuration files
│   ├── default.yaml            # Default settings (chunk_size, models, etc.)
│   └── local.yaml              # Local overrides (gitignored if sensitive)
├── data/                       # Sample documents for testing & demos (gitignored)
├── tests/                      # Unit & integration tests
│   ├── test_extraction.py
│   ├── test_classification.py
│   └── test_rag.py
├── docs/                       # Additional documentation (optional but recommended)
│   └── architecture.md         # High-level design, decisions, future ideas
├── .github/                    # CI/CD automation
│   └── workflows/
│       └── ci.yml              # GitHub Actions: lint, test, etc.
├── .env.example                # Safe template (keys intentionally empty)
├── requirements.txt            # Python dependencies
├── README.md
└── LICENSE
```

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
