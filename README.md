# AI Doc to Chat

> **Privacy · Reproducibility · Honesty** — grounded answers on *your* infrastructure, not a black-box SaaS tab.

[Live Pilot](https://ai-doc-pilot.roxanatapia.dev) · [Public Demo](https://ai-doc-to-chat-demo.streamlit.app) · [Deployment Guide](DEPLOYMENT.md)

**Private RAG chat for your PDFs** — upload contracts, invoices, or scans, ask questions in plain language, and get **grounded answers with page-level sources**. Built for teams that need **control over data and infrastructure**, not another public chatbot tab.

**Clients who care about privacy and control don't upload contracts or HR policies to ChatGPT.** Try the [live pilot](https://ai-doc-pilot.roxanatapia.dev) (password-protected, real local AI) or [hire on Upwork](https://www.upwork.com/freelancers/roxanadev) for a private deployment on your infrastructure.

---

## Three ways to use this project

| Mode | Where | Generation | Best for |
|------|--------|------------|----------|
| **Public demo** | [Streamlit Cloud](https://ai-doc-to-chat-demo.streamlit.app) | UI + retrieval preview (no LLM on that host) | Try the experience, no login required |
| **Reference pilot** | [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev) | **Real Ollama** on a dedicated VPS — HTTPS + basic auth | See a live private deployment; request access to evaluate |
| **Your own deployment** | Your infrastructure | **Real Ollama** — answers grounded in your documents | Confidential PDFs on infrastructure you control |

**Reference pilot disclaimer:** The pilot at `ai-doc-pilot.roxanatapia.dev` is provided for evaluation and demonstration purposes only. Use only sample or non-confidential documents. **Uploaded files are processed in memory and never written to disk or retained between sessions** — each browser reload starts fresh with no stored data. The pilot operator accepts no liability for content submitted by visitors. For confidential documents, [deploy your own instance](DEPLOYMENT.md) on infrastructure you control.

A fictional sample NDA for testing is available as [`docs/sample-nda.pdf`](docs/sample-nda.pdf) (and [`docs/sample-nda.md`](docs/sample-nda.md) for reference).

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
2. **Private pilot** — real local AI on your or my infrastructure, suitable for confidential or redacted documents under NDA. Password-protected HTTPS on a dedicated server; see [For teams and consulting](#for-teams-and-consulting).
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

**Reference deployment** — Docker Compose stack (app + local Ollama) with **HTTPS and basic auth** via Caddy, live at [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev); see the [deployment guide](DEPLOYMENT.md).

**Coming next:** demo video and architecture diagram for sales calls.

---

## Pilot scope & limitations

Honest expectations for the **evaluation pilot** — not a limitation of RAG in general, but of this reference stack today.

**Strong fit**

- Q&A over **contracts, policies, reports, and briefs** — find clauses, dates, parties, obligations
- **Sourced answers** with page-level excerpts (inspect what the model used)
- **Private self-hosted** evaluation under NDA — no documents sent to public chat tools
- **Digital PDFs** and many scans (with optional OCR)

**Known limits**

- **Session-based** — re-upload after restart; no shared team library yet ([Future work](#future-work))
- **Retrieval + read**, not **calculate** — totals printed on the page are fine; summing line items or verifying VAT math is unreliable
- **Tables and invoices** — line items may be partial or jumbled after text extraction; not a substitute for AP automation or spreadsheet validation
- **Chat assistant**, not an **agent** — answers questions; does not call your CRM, email, or ticket systems
- **Single-document focus per session** — not enterprise search across SharePoint, databases, and APIs yet

For production depth (API, persistence, SSO, ops), see [Future work](#future-work). For setup, see [DEPLOYMENT.md](DEPLOYMENT.md).

**Evaluation example:** [Round 5 pilot test](docs/pilot-evaluation-round5.md) on a sample NDA and a client-style 2-page NDA — what was measured, what worked, and current limits.

---

## Future work

The current pilot is a **session-based** app (indexes per upload). Typical production extensions on the same RAG core:

- **API layer** — integrate beyond the Streamlit UI
- **Persistent storage** — documents and vectors survive restarts
- **Enterprise access** — SSO, security documentation, operational runbooks
- **Flexible LLM backend** — local Ollama, or private cloud APIs where procurement requires it

---

## Self-hosted pilot

Follow the [Deployment Guide](DEPLOYMENT.md) for prerequisites, sizing, HTTPS with basic auth, and step-by-step setup on a VPS or your laptop.

---

## For teams and consulting

> A private, password-protected pilot on a dedicated server — upload PDFs, chat with real local AI, documents processed in your environment. Ideal for evaluation before production rollout.

I deploy **private document AI** for organizations that cannot send contracts or policies to public chat tools — from single-VM pilots through production-shaped stacks on **your** cloud or dedicated infrastructure.

| | |
|--|--|
| **Privacy** | Documents and prompts stay in your environment; local or VPC-hosted LLM |
| **Reproducibility** | Docker Compose reference deployment your IT can audit and reproduce |
| **Honesty** | Public demo shows the UI; private pilot delivers real generation — clearly separated |

**Typical path:** evaluation on a [modest dedicated server](DEPLOYMENT.md) → hardening and persistence → production with auth and runbooks. Higher quality and scale mean more compute — either a larger VM you control or a scoped cloud API — chosen with your IT and legal team.

This reference pilot was built with **[Cursor](https://cursor.com)** — the same test-backed, reviewable workflow I use to ship client work without cutting corners. That is how the stack, deployment guide, and docs moved quickly while staying auditable in git. **Your PDFs and prompts are not part of that:** at runtime everything stays on **your** VM or VPC (local Ollama). My tooling builds the product; your data never leaves your environment.

**Get in touch:** [Upwork](https://www.upwork.com/freelancers/roxanadev) (private pilot, consulting, production scoping) · [GitHub](https://github.com/RoxanaTapia) (OSS and technical questions)

---

## Contributing

Pull requests welcome. For how this repo is organized (agents, milestones, verify-before-merge), see [AGENTS.md](AGENTS.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Core stack

Streamlit · LangChain · FAISS · sentence-transformers · PyMuPDF · Tesseract · Ollama · Docker

MIT licensed — free to use, modify, or build on commercially.

Made with ❤️ by **[Roxana Tapia](https://github.com/RoxanaTapia)** — 2026
