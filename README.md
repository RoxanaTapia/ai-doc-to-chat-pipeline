# AI Doc to Chat

> **Privacy · Reproducibility · Honesty** — grounded answers on *your* infrastructure, not a black-box SaaS tab.

[Live Demo](https://ai-doc-to-chat-demo.streamlit.app) · [Deployment Guide](DEPLOYMENT.md) · [Production Roadmap](docs/ROADMAP.md)

**Private RAG chat for your PDFs** — upload contracts, invoices, or scans, ask questions in plain language, and get **grounded answers with page-level sources**. Built for teams that need **control over data and infrastructure**, not another public chatbot tab.

**Want real answers on confidential documents?** The public demo shows the UI only. [Contact me](#for-teams-and-consulting) for a **private pilot** (real Ollama) or to deploy inside your company.

---

## Two ways to use this project

| Mode | Where | Who hosts | Generation | Best for |
|------|--------|-----------|------------|----------|
| **Public demo** | [Streamlit Cloud](https://ai-doc-to-chat-demo.streamlit.app) | Streamlit (free tier) | Dummy — retrieval + UI work; no LLM on that server | Try the UX, share a portfolio link |
| **Private pilot** | Your laptop **or** one VPS / company VM | **You**, your client, or me (consulting) | **Real Ollama** via [Docker Compose](DEPLOYMENT.md) | Confidential PDFs, sales demos, pilots |

Both use the **same app code**. The difference is **where it runs** and whether **real local AI** (`Ollama`) is attached.

### Why laptop *and* VPS?

They are the **same stack**, on different machines:

| Setup | What it is | Typical use |
|-------|------------|-------------|
| **Laptop** | Docker Compose on your Mac/PC — app + Ollama on `localhost:8501` | Build, test, record demos, learn the stack |
| **Single VPS** | Same Compose on one cloud VM (e.g. Hetzner) — team opens `https://your-pilot.example.com` | Client pilot, small-team access, “hire me” reference deployment |

Nothing in the repo forces one or the other. **One `docker-compose.yml`** defines app + Ollama; you run it wherever Docker runs.

---

## How it works (architecture in plain English)

Each question you ask goes through these layers on **one machine** (laptop or VM):

```text
┌─────────────────────────────────────────────────────────────┐
│  Single VM (laptop or cloud server)                         │
│                                                              │
│  Browser ──► Streamlit app (:8501)                           │
│                 │                                            │
│                 ├─► PDF upload → extract text (PyMuPDF/OCR)  │
│                 ├─► Chunk + embed → FAISS search (local)     │
│                 ├─► Retrieve relevant pages (no cloud LLM)     │
│                 └─► Ollama (:11434) → grounded answer        │
│                      ▲                                       │
│                      └── OLLAMA_HOST (Docker network)        │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Role |
|-------|------------|------|
| **UI** | Streamlit | Upload PDF, chat, show sources (page, score, snippet) |
| **Retrieval** | LangChain + FAISS + sentence-transformers | Find the right chunks in *your* document — stays local |
| **Generation** | Ollama | Local LLM writes the answer from retrieved context only |
| **Packaging** | Docker + Compose | One command to run app + Ollama together; same recipe on any VM |

**Today:** indexes live in memory per session (fine for pilots). **Roadmap (M9+):** Postgres/pgvector + file storage so documents survive restarts — see [Production direction](#production-direction-m8m12).

---

## Configuration: “the switches”

These settings control **real vs dummy** answers and **which Ollama** to call. They can come from three places (first match wins for env vars):

| Source | File | When it applies |
|--------|------|-----------------|
| **Compose** | [`docker-compose.yml`](docker-compose.yml) `environment:` | Default for Docker pilot — **no `.env` required** |
| **Optional overrides** | [`.env`](.env.example) (copy from `.env.example`) | Custom model, context size, or non-Compose runs |
| **YAML defaults** | [`configs/config.yaml`](configs/config.yaml) `rag.generation` | Fallback when an env var is empty (e.g. model `llama3.1:8b`) |

**Pilot switches (most important):**

| Switch | Compose value | Meaning |
|--------|---------------|---------|
| `USE_DUMMY_GENERATOR` | `false` | Use **real Ollama** answers (required for private pilot) |
| `OLLAMA_HOST` | `http://ollama:11434` | App talks to the **Ollama container** on Docker’s internal network (`ollama` = service name, not `localhost`) |
| `OLLAMA_MODEL` | `phi3:mini` | Which model to load (CPU-friendly demo; YAML default is `llama3.1:8b`) |

**How the app reads them:** `src/app.py` loads `.env` via `python-dotenv`, then reads `USE_DUMMY_GENERATOR`. `src/rag.py` reads `OLLAMA_*` and merges with `configs/config.yaml`. In Compose, variables are injected by Docker — you do **not** need a `.env` file unless you want overrides.

**DEPLOYMENT table:** the markdown table in [DEPLOYMENT.md § Environment variables](DEPLOYMENT.md#environment-variables) documents the same Compose defaults as `docker-compose.yml` — for humans, not for Docker.

**Dockerfile `ENV` vs Compose `environment`:** the [Dockerfile](Dockerfile) sets **Streamlit/Hugging Face noise reduction** (port, log levels). [Compose](docker-compose.yml) sets **RAG pilot switches** (`OLLAMA_*`, `USE_DUMMY_GENERATOR`). Different jobs; both end up as environment variables inside the container.

---

## Do clients need to clone this repo?

**It depends who hosts:**

| Scenario | What the client does | Who runs Ollama |
|----------|----------------------|-----------------|
| **Public Streamlit demo** | Open the URL — no clone | Nobody (dummy answers only) |
| **You host a private pilot** | You send a **password-protected URL** (VPS + HTTPS, M7-6) | You, on your VM — client never clones |
| **Client self-hosts** | Their IT clones repo, runs Compose on **their** VM | Client — data never leaves their infra |
| **Consulting engagement** | You deploy on their cloud or a dedicated VM they own | Client-owned or contractually isolated |

The repo is a **reference deployment** (copy-paste recipe), not the only delivery model. Many buyers never touch git — they get a URL or you install it for them.

**Without `.env.example` / Compose docs:** you could still run *your* hosted pilot by setting env vars in Compose — but clients and IT would not know the recipe. That documentation is the **portfolio + sales asset**.

---

## For buyers: demo vs full pilot

**Recommended funnel:**

1. **Public demo** — zero friction; proves UX and retrieval. Answers are intentionally **not** from a real LLM (Streamlit Cloud cannot run Ollama for you).
2. **Private pilot** — contact for access: real Ollama, their or your VM, NDA-friendly. This is where **Privacy · Reproducibility · Honesty** matter.
3. **Production** — persistence, SSO, runbooks ([M8–M12](#production-direction-m8m12)).

Showing real private RAG **immediately** to everyone is possible but costly (GPU/CPU, abuse, support). **Gating the full pilot** (email / call / shared URL) is a strong strategy: the public demo **teases**; the pilot **closes**.

---

## How a company deployment typically looks

**Small team pilot (5–20 users, one VM):**

```text
Employees → HTTPS (Caddy + basic auth or SSO later)
              → Streamlit on company VM or VPC
              → Ollama on same VM
              → PDFs processed in-session (M7); stored in DB later (M9)
```

| Phase | What you get |
|-------|----------------|
| **Week 1 — Pilot** | Single VM, Compose, `phi3:mini` or `llama3.1:8b`, password on URL |
| **Month 1–3 — Hardening** | Backups, monitoring, model tuning, optional FastAPI (M8) |
| **Production** | Client-owned cloud, pgvector, SSO, audit trail (M9–M11) |

**Repeatable on any single VM** means: same `git clone` + `docker compose up` works on your laptop, a Hetzner box, or a client’s Azure VM — no “works on my machine” magic. Advantage for a **small team**: one IT person can reproduce the stack from this repo; you help with sizing, HTTPS, and compliance narrative.

Details: [DEPLOYMENT.md](DEPLOYMENT.md) · [docs/architecture-pilot.md](docs/architecture-pilot.md)

---

## What works today

### Application (M1–M6, `v0.6.0` baseline)

- PDF upload with **PyMuPDF** + optional **Tesseract OCR** for scans  
- **Chunking**, **local embeddings** (`all-MiniLM-L6-v2`), **FAISS** retrieval  
- **Semantic** and **hybrid (BM25 + dense)** retrieval with optional reranker  
- **Streamlit** chat UI with source previews (page, score, chunk text)  
- **Ollama** or dummy generation ([`configs/config.yaml`](configs/config.yaml), `.env`)  
- Developer mode: retrieval metrics, exact context fed to the LLM  

### Reference deployment (M7 — in progress)

Shipped on `main` so far:

- [`Dockerfile`](Dockerfile) — reproducible Streamlit image (Python 3.12, OCR runtime)  
- [`docker-compose.yml`](docker-compose.yml) — **app + Ollama**, health-gated startup, persistent model volume  
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — build, compose, model pull, env vars, troubleshooting  
- [`.env.example`](.env.example) — self-host settings (`OLLAMA_HOST`, `USE_DUMMY_GENERATOR`, …)  
- [`AGENTS.md`](AGENTS.md) + [`.cursor/agents/`](.cursor/agents/) — milestone delivery playbook for contributors  

Still open for **M7 finish line** (see [roadmap](docs/ROADMAP.md)): full VPS/HTTPS guide, Caddy, demo video, pilot URL.

---

## Production direction (M8–M12)

The demo is a **Streamlit session app** (in-memory FAISS per upload). Production pilots and enterprise delivery add layers on the same RAG core:

| Milestone | Delivers |
|-----------|----------|
| **M8** | Extract `rag/` package + **FastAPI** (`/health`, `/chat`, OpenAPI) — integration surface beyond Streamlit |
| **M9** | **Postgres/pgvector** + object storage — documents and indexes **survive restart** |
| **M10** | **SSO / auth proxy**, `SECURITY.md` |
| **M11** | **RUNBOOK.md**, backups, monitoring |
| **M12** | **Anthropic** (or Azure OpenAI) as swappable LLM backend; pilot/pricing docs; demo recording |

Target architecture:

```text
Browser → HTTPS → Streamlit and/or FastAPI
                      ├── Postgres + pgvector
                      ├── MinIO (PDFs)
                      └── LLM (Ollama | Anthropic | …)
```

Details: [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/architecture-pilot.md](docs/architecture-pilot.md)

---

## Quick start (self-hosted, real answers)

**Fastest path:** Docker Compose with Ollama (see [DEPLOYMENT.md](DEPLOYMENT.md) for full steps).

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline

docker compose up -d ollama
docker compose exec ollama ollama pull phi3:mini   # once, into persistent volume
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). Compose sets `OLLAMA_HOST=http://ollama:11434` and `USE_DUMMY_GENERATOR=false`.

**Local dev without Compose:** install [Ollama](https://ollama.com), `pip install -r requirements.txt`, `ollama pull llama3.1:8b`, then `streamlit run src/app.py` (dummy mode off in sidebar).

**Streamlit Cloud demo:** [ai-doc-to-chat-demo.streamlit.app](https://ai-doc-to-chat-demo.streamlit.app) — dummy generation only; use self-host for real LLM answers.

---

## Project milestones

| Phase | Status |
|-------|--------|
| M1–M4 | ✅ Prototype, PDF/OCR, FAISS RAG, generation |
| M5–M6 (`v0.6.0`) | ✅ Live demo, cloud-ready shell |
| **M7** | 🚧 Reference deployment (Dockerfile, Compose, health gate — [milestone #1](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/1)) |
| M8–M12 | 📋 Planned — API, persistence, auth, ops, commercial docs |

Last tagged release: **`v0.6.0`**. Post-`v0.6.0` work ships via `main` and GitHub milestones until the next tag.

---

## For teams and consulting

I help organizations deploy **private document AI** — single-VM pilots through production-shaped stacks (Compose, VPS, optional FastAPI, SSO, vector DB), without sending confidential PDFs to public SaaS chat.

**Reference deployment** = this public repo proves: *I don’t only have a demo link — I can ship a private, deployable version.*

| Buyer concern | How this project addresses it |
|---------------|----------------------------------|
| **Privacy** | Documents and prompts stay on *your* VM; Ollama runs locally |
| **Reproducibility** | Docker Compose + documented env — IT can run the same stack |
| **Honesty** | Public demo = UI only; private pilot = real generation, clearly labeled |

**Get in touch for:**

- A **live private demo** (real Ollama on sample or redacted docs)  
- **Pilot setup** on your laptop, VPS, or company cloud  
- **Production path** — persistence, auth, runbooks ([roadmap](docs/ROADMAP.md))

Typical engagement path:

1. **Pilot** — self-hosted Compose + Ollama on your or my infrastructure  
2. **Production** — persistence, auth, runbooks, client-owned cloud or dedicated VM  
3. **Engine choice** — local Ollama for air-gap; Anthropic/Azure OpenAI via private API when quality or procurement requires it  

Same RAG pipeline; deployment and LLM backend vary by compliance and budget.

---

## Development

- Config: [`configs/config.yaml`](configs/config.yaml), [`configs/prompts.yaml`](configs/prompts.yaml)  
- Tests: `pytest tests/ -v`  
- Contributor workflow: [AGENTS.md](AGENTS.md) (orchestrator, slash commands, one-issue-one-PR)  

---

## Core stack

Streamlit · LangChain · FAISS · sentence-transformers · PyMuPDF · Tesseract · Ollama · Docker · YAML-driven config

MIT licensed — free to use, modify, or build on commercially.

Made with ❤️ by **Roxana Tapia** — 2026
