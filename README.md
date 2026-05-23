# AI Doc to Chat

> **Privacy · Reproducibility · Honesty** — grounded answers on *your* infrastructure, not a black-box SaaS tab.

[Live Demo](https://ai-doc-to-chat-demo.streamlit.app) · [Deployment Guide](DEPLOYMENT.md)

**Private RAG chat for your PDFs** — upload contracts, invoices, or scans, ask questions in plain language, and get **grounded answers with page-level sources**. Built for teams that need **control over data and infrastructure**, not another public chatbot tab.

**Clients who care about privacy and control don't upload contracts or HR policies to ChatGPT — or a [public demo](https://ai-doc-to-chat-demo.streamlit.app).** For a **private pilot** with real local AI, [hire on Upwork](https://www.upwork.com/freelancers/roxanadev) or [reach out on GitHub](https://github.com/RoxanaTapia).

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

**Typical path:** evaluation on a [modest dedicated server](DEPLOYMENT.md) → hardening and persistence → production with auth and runbooks. Higher quality and scale mean more compute — either a larger VM you control or a scoped cloud API — chosen with your IT and legal team.

**Get in touch:** [Upwork](https://www.upwork.com/freelancers/roxanadev) (private pilot, consulting, production scoping) · [GitHub](https://github.com/RoxanaTapia) (OSS and technical questions)

---

## Contributing

This project is built **agentically in [Cursor](https://cursor.com)** — one GitHub issue, one branch, one reviewable PR — so pilot features ship in small, tested steps instead of ad-hoc prompts.

**How it works**

| Cursor feature | Role in this repo |
|----------------|-------------------|
| **Rules** (`.cursor/rules/`) | Persistent standards — Python/RAG style, Docker deploy, docs tone, milestone workflow |
| **Subagents** (`.cursor/agents/`) | Specialists with file ownership (`deploy-engineer`, `rag-core-engineer`, `docs-writer`, `verifier`, …) |
| **Slash commands** (`.cursor/commands/`) | Repeatable flows — e.g. `/ship-m7-issue`, `/verify` |
| **Orchestrator** | Reads issues, dispatches specialists, runs tests, splits granular commits |

**Why bother**

- **Clear ownership** — agents edit only their files; fewer merge conflicts and surprise diffs
- **Production-shaped history** — granular commits and PRs that map to roadmap milestones
- **Verify before merge** — pytest/ruff (and Docker checks when infra changes) before every PR
- **Human in the loop** — you approve commits, pushes, and merges; agents propose, you decide

**Get started:** [AGENTS.md](AGENTS.md) (playbook) · [docs/ROADMAP.md](docs/ROADMAP.md) (milestones M7–M12) · open a GitHub issue or pick one from the [M7 milestone](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/1).

Using Cursor is optional for drive-by fixes — but matching the one-issue-one-PR workflow helps reviews stay fast.

---

## Core stack

Streamlit · LangChain · FAISS · sentence-transformers · PyMuPDF · Tesseract · Ollama · Docker

MIT licensed — free to use, modify, or build on commercially.

Made with ❤️ by **[Roxana Tapia](https://github.com/RoxanaTapia)** — 2026
