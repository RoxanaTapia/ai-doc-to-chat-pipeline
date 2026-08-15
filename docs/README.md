# Documentation

## Background

This page is the map of the repository. The [root README](../README.md) is the client shop window. Start here when you want to know where code lives, how an answer is produced, and how work ships.

> **Takeaway:** Product code in `src/` and `configs/`. Ops in `deploy/`. Shipping in `.cursor/` and [AGENTS.md](../AGENTS.md).

---

## Where to find what

```text
.
├── README.md          Client overview and live pilot
├── AGENTS.md          How Cursor agents ship issues
├── DEPLOYMENT.md      Self-host and VPS notes
├── src/               Application code
├── configs/           Chunking, retrieval, and prompt tunables
├── tests/             pytest suite
├── deploy/            Docker, Compose, Caddy
├── .cursor/           Agent roles, rules, slash commands
└── docs/              You are here
      product/         Architecture, demo script, sample NDA
      operators/       Roadmap, direction, repo-structure detail
      archive/         Historical eval notes (not current)
```

| Path | Open it for |
|------|-------------|
| [`src/app.py`](../src/app.py) | Upload, chat UI, hybrid search, rerank, citations |
| [`src/sectioning.py`](../src/sectioning.py) | Header-aware chunking and section routing |
| [`src/retrieval_quality.py`](../src/retrieval_quality.py) | Dedupe, overlap filter, honesty guard |
| [`src/ocr.py`](../src/ocr.py) | Scanned-page OCR |
| [`src/rag/`](../src/rag/) | Answer generation (Ollama or Anthropic) |
| [`src/api/`](../src/api/) | Thin FastAPI `/health` and `/chat` |
| [`configs/config.yaml`](../configs/config.yaml) | Chunk size, hybrid weights, reranker |
| [`configs/prompts.yaml`](../configs/prompts.yaml) | Grounded-answer prompt |
| [`deploy/`](../deploy/) | Container stack and edge proxy |
| [`docs/product/architecture.md`](product/architecture.md) | Single-VM deploy picture |
| [`docs/operators/REPO-STRUCTURE.md`](operators/REPO-STRUCTURE.md) | Extra layout rules |

---

## How an answer is produced

Two layers: the running services, then the Python modules that do the retrieval work.

### Runtime

The browser never talks to the index or the model directly. Caddy terminates HTTPS. Streamlit holds the session (PDF, chunks, FAISS) in memory. The LLM is either Ollama on the same VM or Anthropic when that provider is selected.

```mermaid
flowchart LR
  Browser --> Caddy
  Caddy --> Streamlit
  Streamlit --> FAISS
  Streamlit --> LLM
```

```mermaid
sequenceDiagram
  participant U as Browser
  participant S as Streamlit
  participant I as In-memory index
  participant M as Ollama or Anthropic

  U->>S: Upload PDF
  S->>S: Extract, OCR if needed, chunk
  S->>I: Build FAISS + BM25
  U->>S: Ask a question
  S->>I: Hybrid search, then rerank
  S->>M: Prompt with cited passages
  M-->>S: Answer
  S-->>U: Answer + page excerpts
```

### Code

Most of the pipeline still lives in `src/app.py`. Helpers around it own chunking, quality, and generation. Tunables come from YAML, not hardcoded constants.

```mermaid
flowchart TB
  App["src/app.py · UI + retrieval"]
  OCR["src/ocr.py"]
  Section["src/sectioning.py"]
  Quality["src/retrieval_quality.py"]
  Gen["src/rag/ · Ollama or Anthropic"]
  Cfg["configs/config.yaml"]
  Prompts["configs/prompts.yaml"]

  App --> OCR
  App --> Section
  App --> Quality
  App --> Gen
  Cfg --> App
  Prompts --> Gen
```

| Stage | Module | Role |
|-------|--------|------|
| Ingest | `app.py` + `ocr.py` | Read the PDF. Keep page numbers. OCR scanned pages. |
| Chunk | `sectioning.py` | Split on section headers so clauses do not bleed. |
| Index | `app.py` | Embed locally. Build FAISS and BM25 for this session. |
| Retrieve | `app.py` | Hybrid search (FAISS + BM25, RRF), then `bge-reranker`. |
| Filter | `retrieval_quality.py` | Drop near-duplicates. Refuse when context is too thin. |
| Generate | `src/rag/` | Write the answer from the shortlist only. |
| Show | `app.py` | Page + excerpt under the reply. |

The FastAPI app in `src/api/` exposes the same generation path as `/chat`. It does not replace the Streamlit retrieval loop.

---

## How work ships

This repo includes a Cursor setup that can take a GitHub issue to a pull request without hand-wiring the steps.

| Piece | Role |
|-------|------|
| [`AGENTS.md`](../AGENTS.md) | Playbook: one issue, one branch, one PR |
| [`.cursor/agents/`](../.cursor/agents/) | Specialists (RAG, Streamlit, docs, deploy, verifier) |
| [`.cursor/commands/`](../.cursor/commands/) | `/ship-issue`, `/ship-milestone`, `/verify` |
| [`.cursor/rules/`](../.cursor/rules/) | Always-on coding and docs conventions |

Specialists edit only what they own. They do not commit. The milestone orchestrator commits, opens the PR, and can merge when checks are green. Full queue and human gates: [AGENTS.md](../AGENTS.md). Direction and sequencing: [operators/PROJECT-DIRECTION.md](operators/PROJECT-DIRECTION.md) · [operators/ROADMAP.md](operators/ROADMAP.md).

---

## Docs in this folder

| Folder | What is here |
|--------|----------------|
| [`product/`](product/) | [Architecture](product/architecture.md), [demo storyboard](product/demo-script.md), [sample NDA](product/sample-nda.pdf) |
| [`operators/`](operators/) | [Roadmap](operators/ROADMAP.md), [project direction](operators/PROJECT-DIRECTION.md), [repo structure](operators/REPO-STRUCTURE.md), [OCR testing](operators/testing-ocr.md) |
| [`archive/`](archive/) | Early local-Ollama eval rounds. Historical only. |

Self-host steps stay in [DEPLOYMENT.md](../DEPLOYMENT.md). Live pilot: [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev/).
