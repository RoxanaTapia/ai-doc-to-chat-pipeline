# Production Roadmap (M7–M12)

Roadmap for evolving the AI Doc-to-Chat pipeline from a local/Streamlit Cloud demo into a
**reference deployment** suitable for private pilots and enterprise delivery.

**Streamlit Cloud** remains the free marketing demo (dummy generation). **M7+** targets a
self-hosted single-VM pilot with real Ollama, then production depth (API, persistence, auth, ops).

---

## Milestone overview

| Milestone | Goal | Hire-me proof |
|-----------|------|---------------|
| **M7** | Reference deployment (single-VM pilot) | Live HTTPS demo + `DEPLOYMENT.md` |
| **M8** | RAG core + FastAPI skeleton | OpenAPI `/health`, `/chat` — not Streamlit-only |
| **M9** | Persistent ingestion | PDFs + vectors survive restart |
| **M10** | Access control | SSO / auth proxy + `SECURITY.md` |
| **M11** | Ops & runbook | Backup, monitoring, `RUNBOOK.md` |
| **M12** | Commercial + multi-engine | Anthropic SKU, pricing playbook, demo video |

**React UI:** optional future milestone — only if a client or RFP requires it.

---

## M7 — Reference deployment (single-VM pilot)

**Target:** 1–2 weeks part-time · **5–7 issues** · ship-first priority

| Issue | Outcome |
|-------|---------|
| M7-1 | `Dockerfile` for Streamlit app |
| M7-2 | `docker-compose.yml` (app + Ollama) |
| M7-3 | Ollama health gate / wait-for-ready |
| M7-4 | Self-host env defaults (`.env.example`, dummy off) |
| M7-5 | `DEPLOYMENT.md` (Ollama path, VPS sizing) |
| M7-6 | HTTPS + basic auth (Caddy) |
| M7-7 | Demo script + architecture diagram |

**Definition of done:** Password-protected HTTPS URL on your VPS, real Ollama answers,
README links to deployment guide. **Enables pilot sales.**

GitHub milestone: [M7](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/1)

---

## M8 — RAG core + FastAPI skeleton

Extract framework-agnostic RAG logic; add thin API layer.

- `src/rag/` package (generation, prompts, config)
- `LLMProvider` interface (Ollama, dummy; Anthropic in M12)
- FastAPI: `/health`, `/chat`, OpenAPI at `/docs`
- Streamlit demo unchanged or calling API

**Definition of done:** `curl /health` and `/chat` work; Streamlit still runs.

---

## M9 — Persistent ingestion

- Postgres + pgvector (+ optional MinIO for PDF blobs)
- Ingest pipeline: upload → chunk → store
- Retrieval from DB instead of session-only FAISS
- Backup/restore documented

**Definition of done:** Reboot server → documents still searchable.

---

## M10 — Access control

- SSO or auth proxy (Microsoft Entra ID, Cloudflare Access, or OAuth2 proxy)
- `docs/SECURITY.md` one-pager
- Optional API key for `/chat` integrations

---

## M11 — Ops & runbook

- `RUNBOOK.md` (update, rollback, logs, incidents)
- Health monitoring (Uptime Kuma or cron + alert)
- Backup scripts for DB and object storage
- Structured logging for app/API

---

## M12 — Commercial + multi-engine

- `AnthropicGenerator` + `LLM_PROVIDER` env switch
- `DEPLOYMENT-ANTHROPIC.md` (no Ollama service; secrets via env/Vault)
- `docs/PILOT-TO-PRODUCTION.md` (tiers, timelines, cost bands)
- `docs/SERVICES.md` (what you deploy for clients)
- 5-minute screen recording (Loom/OBS) linked from README

---

## Workflow

### Branches

One branch per issue, one PR per issue:

```text
feat/m7-dockerfile      → closes M7-1
feat/m7-docker-compose  → closes M7-2
```

### Commits per issue

| Issue size | Typical commits |
|------------|-----------------|
| Docs only | 1 |
| Small infra | 1–2 |
| Feature | 2–4 |
| Large (e.g. M9 ingest) | 3–5 |

### Labels

- Milestone: `m7-deploy` … `m12-commercial`
- Type: `type-infra`, `type-docs`, `type-feature`, `type-test`
- Priority: `priority-ship-first` (M7 blockers)

### GitHub Project (manual setup)

1. Repo → **Projects** → New board: `Production roadmap`
2. Columns: `Backlog` | `Ready` | `In progress` | `In review` | `Done`
3. Add all milestone issues; mark M7-1–M7-5 as **Ready**

---

## When to sell

| Stage | You can… |
|-------|----------|
| After **M7** + partial **M12** docs | Sell pilots; show live VPS demo |
| After **M8** | Prove API/production contract |
| After **M9–M11** | Close serious production deals |

---

## Architecture target (end state)

```text
Browser → HTTPS (Caddy) → Streamlit and/or FastAPI
                              │
                              ├── Postgres + pgvector
                              ├── MinIO (PDFs)
                              └── LLM backend (Ollama | Anthropic)
```

Current repo (pre-M7): single Streamlit process, in-memory FAISS, external Ollama.

---

## Cursor workflow (agents & rules)

This repo uses Cursor **rules**, **subagents**, and **slash commands** to deliver M7–M12 with one-issue-one-PR discipline.

| Resource | Location |
|----------|----------|
| Orchestration playbook | [AGENTS.md](../AGENTS.md) |
| Agent definitions | `.cursor/agents/` |
| Slash commands | `/ship-m7-issue`, `/ship-milestone`, `/verify`, `/parallel-m7-bootstrap` |
| Workflow rules | `.cursor/rules/milestone-workflow.mdc` |

**Operator loop:** pick GitHub issue → `/ship-m7-issue #NN` → resolve blockers → approve commits → merge PR.

M7 issues [#33–#39](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/1) map to specialists (`deploy-engineer`, `config-guardian`, `docs-writer`) — see AGENTS.md.
