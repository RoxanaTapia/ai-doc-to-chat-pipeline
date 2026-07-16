# Production roadmap (portfolio-aligned)

## Background

For contributors and operators. Evolves the pipeline from a **reference deployment** into a **hire-me hero demo**: fast cited answers on a walkthrough, then a thin API for proposals. Deep production (persist / SSO / ops) is client-triggered, not the default climb after M7.

> **Takeaway:** Ship M7.8 → video → packaging first. Thin M8 next. Support MVP / n8n CRM is a separate later project.

**Positioning:** private **document Q&A** for confidential PDFs (contracts, policies, SOPs, reports, internal KB exports), not legal-only. LLM backend is swappable: local Ollama (air-gap) or Anthropic/OpenAI (speed for demo and SaaS-style pilots).

**Streamlit Cloud** = UI-only marketing demo (dummy generation). **Live pilot** = real RAG on your VPS. **M7.8** = demo-quality generation so the video is recordable.

---

## 🎯 Portfolio strategy

Three-piece Upwork niche (only the first is this repo’s north star):

| Piece | Repo / project | Role |
|-------|----------------|------|
| **Hero** | **This repo (ai-doc)** | Private RAG / document Q&A with sourced answers; pilot already live |
| **Sibling story** | [receipt-intelligence-n8n](https://github.com/RoxanaTapia/receipt-intelligence-n8n) | n8n + AI workflow automation |
| **Later** | Support MVP (separate project) | Website chat + RAG + escalate + n8n → CRM |

**Out of scope for this repo’s north star**

- Do not redefine ai-doc as a “website customer support assistant.”
- Do not add escalate + n8n CRM leads as core M8+ goals here.
- Support MVP may reuse RAG patterns later; it is not the next milestone train.

---

## 🗺️ Phase map

| Phase | Milestones | You can say to clients / Upwork |
|-------|------------|----------------------------------|
| **0. Shipped** | M7 ✅ | “Reproducible private pilot on one VM; live URL; deployment guide.” |
| **1. Demo & trust** | **M7.8** → video → **portfolio packaging** | “Fast, cited answers on a walkthrough; calm README; clear pilot + Cloud links.” |
| **2. Thin market contract** | **M8** (optional) | “Not Streamlit-only: `/health`, `/chat`, OpenAPI” when jobs ask for it. |
| **2b. Optional next** | **M8.5** eval | “Before/after retrieval report for buyers / RAG audit gigs.” |
| **3. Client-triggered** | M9 → M11 | “Docs survive restart; SSO; runbooks” when a pilot/RFP needs them |
| **4. Light commercial** | M12 | “Pilot tiers / services one-pager” |

```mermaid
flowchart TD
  M7[M7 shipped] --> M78[M7.8 demo tier]
  M78 --> V[Demo video]
  V --> P[Portfolio packaging]
  P --> Pause[Hire-me ready pause]
  Pause --> Support[Support MVP sibling project]
  Pause -.->|optional if jobs ask| M8[Thin M8 API]
  M8 -.-> M85[M8.5 eval optional]
  Pause -.->|client-triggered| M9[M9–M11]
```

**Hire-me ready** after Phase 1 (video + packaging). Thin M8 only if FastAPI keeps showing up in job posts. Then **pause this repo** and start the Support MVP sibling unless a paid client needs more depth here. M8.5 and M9–M12 do not unlock Support MVP.

---

## 📋 Milestone overview

| Milestone | Goal | Hire-me proof | Priority |
|-----------|------|---------------|----------|
| **M7** | Reference deployment | Live HTTPS pilot + `DEPLOYMENT.md` | ✅ Done |
| **M7.8** | Demo-ready tier | Swappable LLM, streaming, recordable walkthrough | **Ship first** |
| **M7-7 / #57** | Demo video | YouTube/Loom linked from README | After M7.8 |
| **Portfolio packaging** | Upwork-ready framing | 16:9 thumbnail, calm README hero, clear links | After video |
| **M8** | Thin FastAPI contract | OpenAPI `/health`, `/chat` | Optional after packaging |
| **M8.5** | Eval harness export | Before/after retrieval report | Optional / secondary |
| **M9** | Persistent ingestion | PDFs + vectors survive restart | Client-triggered |
| **M10** | Access control | SSO / auth proxy + `SECURITY.md` | Client-triggered |
| **M11** | Ops & runbook | Backup, monitoring, `RUNBOOK.md` | Client-triggered |
| **M12** | Light commercial pack | Pilot tiers, `SERVICES.md` one-pager | Light |

**React UI:** optional, only if a client or RFP requires it.

---

## M7: Reference deployment ✅ shipped

| Issue | Outcome | Status |
|-------|---------|--------|
| M7-1 | `Dockerfile` | ✅ #33 |
| M7-2 | `docker-compose.yml` | ✅ #34 |
| M7-3 | Ollama health gate | ✅ #35 |
| M7-4 | `.env.example` | ✅ #36 |
| M7-5 | `DEPLOYMENT.md` | ✅ #37 |
| M7-6 | Caddy HTTPS | ✅ #38 |
| M7-7a | Demo script + architecture | ✅ #39 |

**Live:** [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev)

### Learnings (M7)

- **Deploy ≠ demo.** HTTPS + Compose proves you ship; CPU Ollama (~30–40s/answer) does not sell on video.
- **One issue → one PR** kept infra reviewable for IT buyers.
- **Honest limits** in README build more trust than “zero hallucinations.”

---

## M7.8: Demo-ready tier 🚧 next

**Target:** part-time · **5 issues** · `priority-ship-first`

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M7.8-1 | `LLMProvider` + `LLM_PROVIDER` env (`ollama` \| `anthropic` \| `openai` \| `dummy`) | [#53](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/53) |
| M7.8-2 | Anthropic adapter (Haiku default for demo/recording) | [#54](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/54) |
| M7.8-3 | Streamlit streaming for generation | [#55](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/55) |
| M7.8-4 | Docs: broaden pitch, non-legal sample PDF, demo-script for Claude tier | [#56](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/56) |
| M7.8-5 | Record demo video + README link | [#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/57) |

**Definition of done:** You can record the video in one session without 40s dead air; narration honestly separates “demo tier (API)” from “self-host tier (Ollama).”

### Learnings (M7.8)

| Issue | Read / learn |
|-------|----------------|
| M7.8-1 | How `generate_answer()` in `src/rag.py` becomes a small provider protocol |
| M7.8-2 | How API keys stay in `.env`, never git; same RAG context, different backend |
| M7.8-3 | Streamlit `st.write_stream` or generator pattern |
| M7.8-4 | Product narrative ≠ code; eval corpus (NDA) ≠ market vertical |
| M7.8-5 | Marketing asset; no new Python required |

**Serial order:** #53 → #54 → #55; #56 parallel with #55; #57 after #54–#56.

GitHub milestone: [M7.8](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/7)

---

## Demo video (after M7.8)

Storyboard: [`docs/product/demo-script.md`](../product/demo-script.md). Record using **Anthropic demo tier**; show architecture slide for **Ollama self-host**. Tracked as [#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/57).

### Learnings

- Video sells **retrieval + citations + deploy path**, not raw local inference speed.
- Pre-warm the model; run storyboard questions once before Record.

---

## Portfolio packaging (after video)

| Item | Outcome |
|------|---------|
| Thumbnail story | Calm 16:9 still (upload → cited answer) |
| README hero | Private document Q&A framing; pilot + Cloud links obvious |
| Video link | README Demo video line points at published walkthrough |
| Honest tiers | Demo (API LLM) vs self-host (Ollama) stated once |

**Definition of done:** A cold visitor understands what you sell and where to click in under a minute.

---

## M8: Thin market contract (FastAPI)

Thin integration surface after video + packaging. Enough for proposals. Not a large API rewrite gate.

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M8-1 | Extract `src/rag/` package | [#58](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/58) |
| M8-2 | FastAPI `/health`, `/chat`, OpenAPI at `/docs` | [#59](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/59) |
| M8-3 | Streamlit calls API when `API_BASE_URL` set (optional) | [#60](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/60) |

**Definition of done:** `curl /health` and `/chat` work; Streamlit still runs standalone.

**Serial:** #58 before #59–#60.

GitHub milestone: [M8](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2)

---

## M8.5: Eval harness export (optional / next)

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M8.5-1 | CLI/script: fixed Q-set → JSON/Markdown report | [#61](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/61) |

**Definition of done:** One command reproduces [`pilot-evaluation.md`](../product/pilot-evaluation.md)-style output for any doc in corpus.

---

## M9: Persistent ingestion (client-triggered)

- Postgres + pgvector (+ optional MinIO)
- Ingest: upload → chunk → store; retrieval from DB
- Backup/restore documented

**Definition of done:** Reboot server → documents still searchable.

**When:** A pilot or RFP needs persistence.

---

## M10: Access control (client-triggered)

- SSO or auth proxy; `docs/SECURITY.md`; optional API key for `/chat`

**When:** Enterprise login is in scope for a paid engagement.

---

## M11: Ops & runbook (client-triggered)

- `RUNBOOK.md`, monitoring, backups, structured logging

**When:** Client IT asks how to operate after pilot yes.

---

## M12: Light commercial pack

- Short `docs/SERVICES.md` / pilot tiers one-pager
- Video link maintained in README
- OpenAI docs only if that SKU is offered

**Note:** Anthropic implementation lives in **M7.8**; M12 is light offers and docs.

---

## 💼 When to sell

| Stage | You can… |
|-------|----------|
| **Now (M7)** | RAG retrieval audit; private pilot deploy; Upwork with live URL |
| **After M7.8 + video + packaging** | Confident walkthrough; portfolio-ready hero |
| **After thin M8** | Proposals that mention `/health` + `/chat` / OpenAPI |
| **After M8.5** | RAG audit / eval-harness gigs with reproducible reports |
| **After M9–M11 (client-scoped)** | Serious production contracts |

---

## 🏛️ Architecture target

```text
Browser → HTTPS (Caddy) → Streamlit and/or FastAPI
                              │
                              ├── Postgres + pgvector (M9, when client needs)
                              ├── MinIO (PDFs, optional)
                              └── LLM (Ollama | Anthropic | OpenAI)
```

**Today:** M7 shipped (Streamlit + Ollama + FAISS per session). **Next:** M7.8 swappable LLM + streaming → video → packaging → thin M8.

Client-readable diagram: [product/architecture.md](../product/architecture.md).

---

## 🛠️ Cursor workflow

| Resource | Location |
|----------|----------|
| Operator guide | [PROJECT-DIRECTION.md](PROJECT-DIRECTION.md) |
| Docs index | [docs/README.md](../README.md) |
| Orchestration | [AGENTS.md](../../AGENTS.md) |
| Commands | `/ship-issue`, `/ship-milestone`, `/verify` |
| Rules | `.cursor/rules/milestone-workflow.mdc` |

**Loop:** pick GitHub issue → **new Agent chat** → `/ship-issue #NN` → review code yourself → `commit` → merge → next issue.

Issue map: M7.8 [#53–#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/7) · M8 [#58–#60](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2) · M8.5 [#61](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2)
