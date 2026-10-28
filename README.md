# PDF Q&A with page citations

> Grounded answers from a confidential PDF, with page citations.

Upload a policy, SOP, report, or contract. Ask in plain language. Every answer cites the page it used.

> **Takeaway:** This is a retrieval system, not a chatbot that guesses. If the document cannot answer, it says so.

---

## Try the pilot

🟢 **Live** · <a href="https://ai-doc-pilot.roxanatapia.dev/" target="_blank" rel="noopener noreferrer"><strong>ai-doc-pilot.roxanatapia.dev</strong></a>

- Request an invite on the gate, then upload a PDF and ask a question
- Start with the [sample NDA](docs/product/sample-nda.pdf) if you want a ready document
- Optional: [ask for a walkthrough on Upwork](https://www.upwork.com/freelancers/roxanadev)

Your PDF and search stay on this server for this browser session; only the few passages needed for an answer are sent to a hosted model (Anthropic). Cleared when you leave or reset. Each session starts fresh. Use only sample or non-confidential documents on the shared pilot.

On your own server, it can run fully offline with a local model.

---

## How an answer is grounded

Each step exists to keep the answer inside the PDF you uploaded.

![Upload a PDF, split by page, search on the server, send passages to the model, then answer with a page quote](docs/assets/aidoc_architecture.png)

| Step | Technique | Why? |
|------|-----------|------|
| **Ingest** | PyMuPDF, plus OCR on scanned pages | PDFs mix digital text and scans. Page numbers stay on the passage so a citation is checkable. |
| **Chunk** | Split at section headers | A sliding window glues neighboring clauses. Headers are the document's own boundaries. |
| **Embed** | Local sentence-transformers | Search by meaning without sending the file to an outside embedding API. |
| **Retrieve** | Hybrid search: FAISS + BM25, fused with RRF | Embeddings miss exact terms. Keyword search misses paraphrases. Fusion keeps both. |
| **Rerank** | Cross-encoder (`bge-reranker`) | First-stage ranking is approximate. Re-score question and passage together before the LLM sees them. |
| **Cite or refuse** | Page + excerpt, or "not in the document" | An answer you cannot open on a page is not grounded. No evidence means no invented clause. |

Each step is a module under [`src/ai_doc/rag/`](src/ai_doc/rag/). Map: [docs/README.md](docs/README.md).

---

## What it does well

| Feature | In practice |
|---------|-------------|
| **Citations** | Page and excerpt on every answer |
| **Model choice** | Ollama writes answers on your server. Anthropic is optional when you want quicker replies. Search and citations stay the same. |
| **On-server embeddings** | Vectors stay on your server |
| **Honest refusals** | Says so when the PDF does not contain the answer |
| **Document types** | Policies, SOPs, reports, contracts |

## Known limits

| Limit | In practice |
|-------|-------------|
| **Session only** | Re-upload after a restart; no shared library yet |
| **One PDF** | Not search across a file store |
| **No math** | Finds printed numbers; does not calculate |
| **Q&A only** | No CRM, email, or ticketing |

---

## What is in this repo

The retrieval core and the Streamlit demo are separate. You can install the core on its own.

| Piece | Path | What it is |
|-------|------|------------|
| **Core** | [`src/ai_doc`](src/ai_doc/) | Python package `ai-doc`. Ingest, chunk, search, citations, and the answer. Optional HTTP API in [`src/ai_doc/api`](src/ai_doc/api/). |
| **Demo** | [`apps/demo`](apps/demo/) | Streamlit client: upload a PDF, ask, and show the page. The core does not import it. |

| Path | What it is |
|------|------------|
| [`configs/`](configs/) | Chunk size, retrieval, and prompts |
| [`deploy/`](deploy/) | App image and Compose: the demo, Ollama, and an optional API |
| [`docs/`](docs/README.md) | Product and operator pages |
| [`.env.example`](.env.example) | Product knobs. Copy to `.env`. Never commit a filled file. |

Module map: [docs/README.md](docs/README.md). Tests and review: [AGENTS.md](AGENTS.md).

---

## Quickstart

Python 3.12 or newer. Clone the repository, then copy the example env file:

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
cp .env.example .env
```

Compose runs the demo and Ollama together. Local Python runs the demo in a virtual environment. For that path, set `OLLAMA_HOST` in `.env` to an Ollama you can reach. The comments in `.env.example` list the usual values.

### Docker Compose

An empty `OLLAMA_MODEL` selects `llama3.1:8b`. Append `phi3:mini` and pass `--env-file .env` so the model you pull is the one the app calls. A value only in the shell is not enough.

```bash
printf '\nOLLAMA_MODEL=phi3:mini\n' >> .env
docker compose --env-file .env -f deploy/docker-compose.yml up --build -d
docker compose --env-file .env -f deploy/docker-compose.yml exec ollama ollama pull phi3:mini
```

The demo listens on port **8501**, path **`/app`**.

Stop the stack:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml down
```

Optional API and other run modes: [docs/operators/running.md](docs/operators/running.md).

### Local Python

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run apps/demo/app.py
```

On Windows, activate with `.venv\Scripts\activate`. The same port and path apply.

---

## Architecture

The browser talks only to the app. The app keeps the PDF, the chunks, and the search index for the session. Embeddings run with the app. A model writes the answer from the passages it was given.

| Piece | Role |
|-------|------|
| **Demo** | [`apps/demo`](apps/demo/) holds the session and the page you see |
| **Core** | [`src/ai_doc`](src/ai_doc/) ingests, searches, cites, and calls the model |
| **Model** | Ollama beside the app, or Anthropic when you select it |

Search and citations stay the same when you switch the model. Retrieval steps: [How an answer is grounded](#how-an-answer-is-grounded). Request flow: [docs/product/architecture.md](docs/product/architecture.md).

Restart the app and you upload again. Nothing is written to a document library.

---

## Pin a version

The core package is `ai-doc` **1.0.0** ([`pyproject.toml`](pyproject.toml), and `ai_doc.__version__`). The demo is not its own package. Streamlit is the `demo` extra.

Release tags are named `vX.Y.Z`. Pin this release with:

```bash
pip install "git+https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline@v1.0.0"
```

Add the demo extra when you need the Streamlit demo:

```bash
pip install "ai-doc[demo] @ git+https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline@v1.0.0"
```

From a checkout of that tag, `pip install .` installs the core, and `python -c "import ai_doc; print(ai_doc.__version__)"` prints `1.0.0`.

---

## Support

Questions are welcome.

| Need | Where |
|------|--------|
| **Bugs and product questions** | [Open an issue](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues) |
| **A walkthrough of the pilot** | [Upwork](https://www.upwork.com/freelancers/roxanadev), also linked under [Try the pilot](#try-the-pilot) |
| **How changes are reviewed** | [How I work with AI](docs/how-we-work.md) |

The core uses semantic versions. Fixes go to the latest minor version of `ai-doc`. Pin an older commit when that exact tree should stay put.

---

## Stack

PyMuPDF · FAISS · BM25 · bge-reranker · sentence-transformers · Streamlit · FastAPI · Ollama or Anthropic

MIT licensed · [Roxana Tapia](https://github.com/RoxanaTapia) · [Upwork](https://www.upwork.com/freelancers/roxanadev) · [How I work with AI](docs/how-we-work.md)
