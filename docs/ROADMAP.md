# Production Roadmap (M7 → M12, portfolio-aligned)

Evolve the pipeline from a **reference deployment** into a **hire-me hero demo** —
fast cited answers on a walkthrough, thin API for proposals — without treating deep
production (persist / SSO / ops) as the default climb after M7.

**Positioning:** private **document Q&A** for confidential PDFs (contracts, policies, SOPs,
reports, internal KB exports) — not “legal-only.” **LLM backend is swappable:** local Ollama
(air-gap) or Anthropic/OpenAI (speed + quality for demo and SaaS-style pilots).

**Streamlit Cloud** = UI-only marketing demo (dummy generation). **Live pilot** = real RAG on
your VPS. **M7.8** = demo-quality generation so the video is recordable.

---

## Portfolio strategy

Three-piece Upwork niche (only the first is this repo’s north star):

| Piece | Repo / project | Role |
|-------|----------------|------|
| **Hero** | **This repo (ai-doc)** | Private RAG / document Q&A with sourced answers; pilot already live |
| **Sibling story** | [receipt-intelligence-n8n](https://github.com/RoxanaTapia/receipt-intelligence-n8n) | n8n + AI workflow automation (document → structured data → handoff) |
| **Later** | Support MVP (separate project) | Website chat + RAG + escalate + n8n → CRM |

**Out of scope for this repo’s north star**

- Do **not** redefine ai-doc as a “website customer support assistant.”
- Do **not** add escalate + n8n CRM leads as core M8+ goals here.
- Support MVP may **reuse RAG patterns later**; it is not the next milestone train.

Upwork “RAG support + n8n” jobs are won by a polished hero demo (fast answers, citations,
clear README/pilot), then a thin FastAPI surface. Deep production is **client-triggered**.

---

## Phase map (what to complete, in order)

| Phase | Milestones | You can say to clients / Upwork |
|-------|------------|----------------------------------|
| **0 — Shipped** | M7 ✅ | “Reproducible private pilot on one VM; live URL; deployment guide.” |
| **1 — Demo & trust** | **M7.8** → video → **portfolio packaging** | “Fast, cited answers on a walkthrough; calm README; clear pilot + Cloud links.” |
| **2 — Thin market contract** | **M8** (thin) | “Not Streamlit-only — `/health`, `/chat`, OpenAPI.” |
| **2b — Optional next** | **M8.5** eval | “Before/after retrieval report for buyers / RAG audit gigs.” |
| **3 — Client-triggered** | M9 → M11 | “Docs survive restart; SSO; runbooks” — when a pilot/RFP needs them |
| **4 — Light commercial** | M12 | “Pilot tiers / services one-pager” — not a huge packaging train |

```text
M7 ✅ → M7.8 (demo tier) → video → portfolio packaging
     → thin M8 (/health + /chat)
     → M8.5 eval (optional / next)
     → M9–M11 when a pilot/client needs them
     → M12 light commercial pack
```

**Hire-me ready** after Phase 1 (+ thin M8 when you want API proof). M9–M12 are not required
for portfolio readiness.

---

## Milestone overview

| Milestone | Goal | Hire-me proof | Priority |
|-----------|------|---------------|----------|
| **M7** | Reference deployment | Live HTTPS pilot + `DEPLOYMENT.md` | ✅ Done |
| **M7.8** | **Demo-ready tier** | Swappable LLM, streaming, recordable walkthrough | **Ship first** |
| **M7-7 / #57** | Demo video | YouTube/Loom linked from README | After M7.8 |
| **Portfolio packaging** | Upwork-ready framing | 16:9 thumbnail story, calm README hero, clear pilot + Cloud links | After video |
| **M8** | **Thin** FastAPI contract | OpenAPI `/health`, `/chat` (optional Streamlit→API) | After packaging |
| **M8.5** | Eval harness export | Before/after retrieval report | Optional / secondary |
| **M9** | Persistent ingestion | PDFs + vectors survive restart | Client-triggered |
| **M10** | Access control | SSO / auth proxy + `SECURITY.md` | Client-triggered |
| **M11** | Ops & runbook | Backup, monitoring, `RUNBOOK.md` | Client-triggered |
| **M12** | Light commercial pack | Pilot tiers, `SERVICES.md` one-pager | Light; not a mega-train |

**React UI:** optional — only if a client or RFP requires it.

---

## M7 — Reference deployment ✅ shipped

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

## M7.8 — Demo-ready tier 🚧 next

**Target:** part-time · **5 issues** · `priority-ship-first`

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M7.8-1 | `LLMProvider` + `LLM_PROVIDER` env (`ollama` \| `anthropic` \| `openai` \| `dummy`) | [#53](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/53) |
| M7.8-2 | Anthropic adapter (Haiku default for demo/recording) | [#54](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/54) |
| M7.8-3 | Streamlit streaming for generation (feels responsive on video) | [#55](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/55) |
| M7.8-4 | Docs: broaden pitch, non-legal sample PDF, demo-script for Claude tier | [#56](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/56) |
| M7.8-5 | Record demo video + README link | [#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/57) |

**Definition of done:** You can record the video in one session without 40s dead air; narration
honestly separates “demo tier (API)” from “self-host tier (Ollama).”

### Learnings (M7.8) — what you should understand in code

| Issue | Read / learn |
|-------|----------------|
| M7.8-1 | How `generate_answer()` in `src/rag.py` becomes a small provider protocol — **duck typing, one entry point** |
| M7.8-2 | How API keys stay in `.env`, never git; same RAG context, different backend |
| M7.8-3 | Streamlit `st.write_stream` or generator pattern — UX separate from model |
| M7.8-4 | Product narrative ≠ code; eval corpus (NDA) ≠ market vertical |
| M7.8-5 | Marketing asset; no new Python required |

**Serial order:** #53 → #54 → #55 (same files: `rag.py`, `app.py`); #56 parallel with #55; #57 after #54–#56.

GitHub milestone: [M7.8](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/7)

---

## Demo video (after M7.8)

Storyboard: [`docs/demo-script.md`](demo-script.md). Record using **Anthropic demo tier**;
show architecture slide for **Ollama self-host** option. Tracked as [#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/57).

### Learnings

- Video sells **retrieval + citations + deploy path**, not raw local inference speed.
- Pre-warm model / run storyboard questions once before Record.

---

## Portfolio packaging (after video)

High-leverage checklist — not a large engineering milestone. Do this before deep API work
if Upwork proposals are the near-term goal.

| Item | Outcome |
|------|---------|
| Thumbnail story | Calm 16:9 still or frame from the walkthrough (upload → cited answer) |
| README hero | Private document Q&A framing; pilot + Cloud demo links obvious in first screen |
| Video link | README Demo video line points at published walkthrough |
| Honest tiers | Demo (API LLM) vs self-host (Ollama) stated once, without apology |

**Definition of done:** A cold visitor can understand what you sell and where to click in under a minute.

---

## M8 — Thin market contract (FastAPI)

Keep M8, but frame it as a **thin** integration surface after video + packaging — enough for
proposals. Do **not** treat a large API rewrite as a gate before “hire-me ready.”

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M8-1 | Extract `src/rag/` package (retrieval, prompts, providers) | [#58](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/58) |
| M8-2 | FastAPI `/health`, `/chat`, OpenAPI at `/docs` | [#59](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/59) |
| M8-3 | Streamlit calls API when `API_BASE_URL` set (optional) | [#60](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/60) |

**Definition of done:** `curl /health` and `/chat` work; Streamlit still runs standalone.

### Learnings (M8)

- **Separation of concerns:** UI (Streamlit) vs thin integration surface (FastAPI).
- **Same providers** from M7.8 — no second LLM integration path.
- Package extract (#58) supports clarity; keep scope modest.

**Serial:** #58 before #59–#60.

GitHub milestone: [M8](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2)

---

## M8.5 — Eval harness export (optional / next)

Secondary for Upwork after demo trust + thin M8. Strong for “debug my RAG” gigs.

| Issue | Outcome | GitHub |
|-------|---------|--------|
| M8.5-1 | CLI or script: fixed Q-set → JSON/Markdown report (chunks, scores, pass/fail) | [#61](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/61) |

**Definition of done:** One command reproduces `pilot-evaluation.md`-style output for any doc in corpus.

### Learnings (M8.5)

- **Eval is a product** for RAG audit gigs — not required before first proposals.
- Prefer after thin M8 so the harness shares the same `src/rag/` package.

---

## M9 — Persistent ingestion (client-triggered)

- Postgres + pgvector (+ optional MinIO)
- Ingest: upload → chunk → store; retrieval from DB
- Backup/restore documented

**Definition of done:** Reboot server → documents still searchable.

**When:** A pilot or RFP needs persistence — not the default path after M8.

### Learnings

- Session FAISS is fine for **pilot**; production buyers need persistence.
- Embedding model version must match at ingest and query.

---

## M10 — Access control (client-triggered)

- SSO or auth proxy; `docs/SECURITY.md`; optional API key for `/chat`

**When:** Enterprise login is in scope for a paid engagement.

### Learnings

- Password + basic auth (M7) is enough for **evaluation**; enterprise needs identity.

---

## M11 — Ops & runbook (client-triggered)

- `RUNBOOK.md`, monitoring, backups, structured logging

**When:** Client IT asks how to operate after pilot yes.

### Learnings

- “Reference deployment” becomes **operable** — scoped to the engagement, not portfolio vanity.

---

## M12 — Light commercial pack

- Short `docs/SERVICES.md` / pilot tiers one-pager (and optional `PILOT-TO-PRODUCTION.md`)
- Video link maintained in README
- OpenAI docs only if that SKU is offered (`DEPLOYMENT-OPENAI.md`)

**Note:** Anthropic implementation lives in **M7.8**; M12 is **light offers and docs**, not a
multi-engine engineering train.

### Learnings

- Sell **engagement tiers** (eval → pilot → production), not a single binary product.

---

## When to sell

| Stage | You can… |
|-------|----------|
| **Now (M7)** | RAG retrieval audit; private pilot deploy; Upwork with live URL |
| **After M7.8 + video + packaging** | Confident walkthrough; portfolio-ready hero |
| **After thin M8** | Proposals that mention `/health` + `/chat` / OpenAPI |
| **After M8.5** | RAG audit / eval-harness gigs with reproducible reports |
| **After M9–M11 (client-scoped)** | Serious production contracts |

---

## Architecture target

```text
Browser → HTTPS (Caddy) → Streamlit and/or FastAPI
                              │
                              ├── Postgres + pgvector (M9, when client needs)
                              ├── MinIO (PDFs, optional)
                              └── LLM (Ollama | Anthropic | OpenAI)
```

**Today:** M7 shipped — Streamlit + Ollama + FAISS per session. **Next:** M7.8 swappable LLM +
streaming → video → portfolio packaging → thin M8.

---

## Cursor workflow

| Resource | Location |
|----------|----------|
| Operator guide (learnings + how to proceed) | [docs/PROJECT-DIRECTION.md](PROJECT-DIRECTION.md) |
| Orchestration | [AGENTS.md](../AGENTS.md) |
| Commands | `/ship-issue`, `/ship-milestone`, `/verify` |
| Rules | `.cursor/rules/milestone-workflow.mdc` |

**Loop:** pick GitHub issue → **new Agent chat** → `/ship-issue #NN` → review code yourself → `commit` → merge → next issue.

Issue map: M7.8 [#53–#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/7) · M8 [#58–#60](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2) · M8.5 [#61](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2)
