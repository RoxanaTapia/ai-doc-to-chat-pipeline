# AI Doc to Chat

> **Privacy · Reproducibility · Honesty** — grounded answers on *your* infrastructure, not a black-box SaaS tab.

[Live Demo](https://ai-doc-to-chat-demo.streamlit.app) · [Deployment Guide](DEPLOYMENT.md)

**Private RAG chat for your PDFs** — upload contracts, invoices, or scans, ask questions in plain language, and get **grounded answers with page-level sources**. Built for teams that need **control over data and infrastructure**, not another public chatbot tab.

**Want real answers on confidential documents?** The [public demo](https://ai-doc-to-chat-demo.streamlit.app) shows the interface only. [Contact me](#for-teams-and-consulting) for a **private pilot** with real local AI, or deployment inside your company.

---

## Two ways to use this project

| Mode | Where | Generation | Best for |
|------|--------|------------|----------|
| **Public demo** | [Streamlit Cloud](https://ai-doc-to-chat-demo.streamlit.app) | UI + retrieval preview (no LLM on that host) | Try the experience, share a link |
| **Private pilot** | Your infrastructure or mine | **Real Ollama** — answers grounded in your documents | Confidential PDFs, evaluations, rollouts |

---

## How it works

Each question flows through a **single private stack** — your documents never leave your environment:

```text
Browser ──► Streamlit (upload + chat)
              ├─► Extract text from PDF (digital + scanned/OCR)
              ├─► Find relevant passages (local embeddings + search)
              └─► Ollama (local LLM) → answer with page-level sources
```

| Layer | Role |
|-------|------|
| **Interface** | Upload PDFs, chat, inspect sources (page, score, excerpt) |
| **Retrieval** | Semantic and hybrid search over your document — runs locally |
| **Generation** | Local LLM produces answers from retrieved context only |
| **Deployment** | Docker Compose packages app + Ollama for a one-VM pilot |

Pilots use in-session indexing (ideal for evaluations). Longer-term deployments add persistent storage, API access, and enterprise auth — see [Future work](#future-work).

---

## Demo vs private pilot

1. **Public demo** — open the link, upload a sample PDF, explore the UX. Generation is intentionally limited on the hosted demo.
2. **Private pilot** — real local AI on your or my infrastructure, suitable for confidential or redacted documents under NDA.
3. **Production rollout** — persistence, SSO, runbooks, and your choice of cloud or on-prem — scoped per engagement.

---

## Typical company pilot

```text
Team ──► HTTPS ──► Streamlit + Ollama (one VM or VPC)
                      └── PDFs processed in your environment
```

| Phase | Outcome |
|-------|---------|
| **Pilot** | Single VM, real answers, access control on the URL |
| **Hardening** | Backups, monitoring, model tuning |
| **Production** | Client-owned cloud, persistent indexes, SSO, audit trail |

Deployment and architecture: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## What works today

**Document AI (stable baseline, `v0.6.0`)**

- PDF upload with **PyMuPDF** and optional **Tesseract OCR** for scans
- Local **embeddings** and **FAISS** retrieval — semantic and hybrid (BM25 + dense)
- **Streamlit** chat with source previews (page, score, chunk text)
- **Ollama** for private generation, or dummy mode for the public demo
- Optional developer view: retrieval metrics and context transparency

**Reference deployment** — Docker Compose stack (app + local Ollama) with a full [deployment guide](DEPLOYMENT.md) for VPS or laptop pilots.

**Coming next:** HTTPS + access control on the pilot URL, demo video.

---

## Future work

The current pilot is a **session-based** app (indexes per upload). Typical production extensions on the same RAG core:

- **API layer** — integrate beyond the Streamlit UI
- **Persistent storage** — documents and vectors survive restarts
- **Enterprise access** — SSO, security documentation, operational runbooks
- **Flexible LLM backend** — local Ollama, or private cloud APIs where procurement requires it

---

## Self-hosted pilot

Follow the [Deployment Guide](DEPLOYMENT.md) for prerequisites, sizing, and step-by-step setup on a VPS or your laptop.

---

## For teams and consulting

I deploy **private document AI** for organizations that cannot send contracts or policies to public chat tools — from single-VM pilots through production-shaped stacks on **your** cloud or dedicated infrastructure.

| | |
|--|--|
| **Privacy** | Documents and prompts stay in your environment; local or VPC-hosted LLM |
| **Reproducibility** | Docker Compose reference deployment your IT can audit and reproduce |
| **Honesty** | Public demo shows the UI; private pilot delivers real generation — clearly separated |

**Typical path:** pilot on your infrastructure → hardening and persistence → production with auth and runbooks. Local Ollama for air-gap; private cloud APIs when quality or procurement requires it.

**[Contact me](#for-teams-and-consulting)** for a live private demo, pilot setup, or scoping a production rollout.

---

## Contributing

Developed in small, tested increments so pilots can harden into production without surprises.

Open-source contributors: [AGENTS.md](AGENTS.md) · [docs/ROADMAP.md](docs/ROADMAP.md)

---

## Core stack

Streamlit · LangChain · FAISS · sentence-transformers · PyMuPDF · Tesseract · Ollama · Docker

MIT licensed — free to use, modify, or build on commercially.

Made with ❤️ by **Roxana Tapia** — 2026
