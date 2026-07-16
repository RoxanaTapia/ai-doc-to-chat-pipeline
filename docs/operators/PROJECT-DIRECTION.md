# Project direction (operator guide)

## Background

How to finish this pipeline with Cursor discipline, code you understand, and a private document Q&A demo you can show without apologizing for latency.

Full milestone detail: [ROADMAP.md](ROADMAP.md) · Agent playbook: [AGENTS.md](../../AGENTS.md) · Docs index: [docs/README.md](../README.md)

> **Takeaway:** Start Phase 1 at #53. Deep production waits for a real client.

---

## 🗺️ Milestone recap

| Milestone | Goal | Status |
|-----------|------|--------|
| **M7** | Reference deployment | ✅ Shipped |
| **M7.8** | Demo-ready LLM tier + streaming | 🚧 Next (#53–#57) |
| **Video + packaging** | Walkthrough + calm README framing | After M7.8 |
| **M8** | Thin `/health` + `/chat` API | After packaging (#58–#60) |
| **M8.5 / M9–M12** | Eval export; persist; SSO; ops; light pack | Optional / client-triggered |

---

## 🎯 Product scope (read first)

| Piece | Where | Role |
|-------|-------|------|
| **This product** | **This repo** | Private RAG / document Q&A with sourced answers |
| **Sibling** | receipt-intelligence-n8n | n8n + AI document→data workflows (separate repo) |
| **Later** | Support MVP (new project) | Website chat + escalate + n8n → CRM |

This repo’s north star is **not** a website support assistant. Escalate/CRM belongs in the Support MVP later. Deep production (M9–M11) is **client-triggered**.

---

## ⭐ North star

> Upload a confidential PDF, ask a question, get a **fast** answer with **page citations** on **your** infrastructure. Optionally expose a **thin** `/health` + `/chat` API for integrations; keep eval export and persistence for when a buyer asks.

If that sentence feels true after **demo + video + packaging + thin M8**, this phase of the repo is complete. Then pause here and start the Support MVP sibling unless a paid client needs more depth.

---

## 🧭 Three objectives

### 1. Cursor best practices

| Rule | Why |
|------|-----|
| **One GitHub issue → one branch → one PR** | Clean history; each merge is explainable |
| **Orchestrator commits; specialists don’t** | Train mode: orchestrator also pushes/merges when green (see AGENTS.md) |
| **Use `/ship-issue #NN`** | Prefer one conductor chat for the delivery train; split if context grows |
| **`/verify` before PR** | pytest; skip full `docker build` unless infra changed |
| **PR starts with `## Main contribution`** | Outcome first, not a file list |

**Branch names**

```text
feat/m7-8-llm-provider     → closes #53
feat/m7-8-anthropic        → closes #54
feat/m8-fastapi-chat       → closes #59
```

**When stuck:** invoke `blocker-reporter` format in AGENTS.md; update Human decisions log; STOP.

**Do not:** one mega-PR for multiple issues; parallel edits on `src/rag.py` + `src/app.py` across two issues.

---

### 2. Learn the code

After each issue, you should answer **without opening Cursor**:

| Layer | Files | Question to answer |
|-------|-------|-------------------|
| **UI** | `src/app.py` | What happens on upload → index → chat? |
| **RAG** | `src/rag.py` → `src/rag/` (M8) | How does context get built before the LLM? |
| **Config** | `configs/config.yaml`, `.env` | What knob changes retrieval vs generation? |
| **Deploy** | `docker-compose*.yml`, `DEPLOYMENT.md` | How does a request reach Ollama or API LLM? |

**Per-issue learning habit (15 min after merge, optional)**

1. Read the diff yourself.
2. Run the app locally: one upload, one question, dev sidebar on.
3. Add **one sentence** to your personal notes (`docs-private/`): what changed and why.
4. If you can’t explain it, open a **question-only** chat.

**Red flag:** merging PRs you don’t understand “to keep speed.” Slow down one issue instead.

---

### 3. Finish line (demo-ready + thin API)

| Phase | Milestones | “I’d use it / I’d show it” test |
|-------|------------|----------------------------------|
| **0** | M7 ✅ | You trust deploy; you don’t trust speed on VPS Ollama |
| **1** | M7.8 → video → packaging | You’d show a colleague the video; README looks calm |
| **2** | Thin M8 | You’d call `/chat` from curl in a proposal |
| **2b** | M8.5 (optional) | You’d send an eval report for a retrieval audit |
| **3** | M9–M11 | Client-triggered |
| **4** | M12 light | Short tiers/services one-pager |

---

## 📅 Phases

### Phase 0: Reference deploy ✅

**Milestones:** M7 (#33–#39)

**Shipped:** Docker, Compose, Caddy, live pilot, DEPLOYMENT, architecture, demo storyboard.

**Learnings:** Retrieval + citations are the core IP. CPU Ollama is an option, not the walkthrough default.

**Your action:** Close M7 on the board. Start Phase 1.

---

### Phase 1: Demo, video & packaging 🚧 START HERE

**Milestones:** M7.8 ✅ → M7.9 ✅ → **M7.95 Sources trust** ([#80–#83](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/9)) → video (#57) → packaging checklist in [ROADMAP.md](ROADMAP.md)

| # | Issue | Agent map | You learn |
|---|-------|-----------|-----------|
| 53–56 | M7.8 demo tier | (shipped) | Provider switch; streaming; pitch |
| 70–73 | M7.9 UI polish | (shipped) | Calm client UI |
| 80 | Sources display cap | config-guardian, streamlit-engineer | Display vs retrieval `top_k` |
| 81 | Sort Sources | rag-core-engineer, streamlit-engineer | On-section + score ordering |
| 82 | Answer-overlap filter | rag-core-engineer, streamlit-engineer | Honest citations with fallback |
| 83 | Header-aware chunking | rag-core-engineer, config-guardian | TOC bleed vs section bodies |
| 57 | Record video + README link | docs-writer (you record) | Walkthrough asset |

```mermaid
flowchart LR
  A["M7.8 + M7.9"] --> B["#80 ∥ #83"]
  B --> C["#81"]
  C --> D["#82"]
  D --> E["#57 record"]
  E --> F[Packaging]
  F --> G[Thin M8]
  G --> H[Phase pause]
```

After packaging + thin M8: **pause** for Support MVP unless a paid engagement needs more here.

**Env for recording (local, never commit)**

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
# optional: ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

---

### Phase 2: Thin API contract

**Milestones:** M8 ([#58–#60](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/2))

Serial #58 → #59 → #60. Included in the default delivery train after packaging.

---

### Phase 2b: Eval export (optional / next)

**Milestone:** M8.5 ([#61](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/61))

---

### Phase 3: Production depth (client-triggered)

**Milestones:** M9, M10, M11 (issues TBD via `/ship-milestone M9`)

Not required for the demo-ready pause.

---

### Phase 4: Light services pack

**Milestone:** M12. Short services/tiers one-pager.

---

## 📆 Weekly rhythm (part-time)

| Day | Activity |
|-----|----------|
| **Mon** | Pick one issue; move to In progress |
| **Tue–Wed** | `/ship-issue` (or continue delivery train); understand diff |
| **Thu** | `/verify`; PR; self-review |
| **Fri** | Merge (or confirm train merge); optional learning notes; update board |

**One issue per week** is enough for M7.8 in about 5 weeks including video.

---

## 📌 GitHub Project board

Columns: `Backlog` | `Ready` | `In progress` | `In review` | `Done`

**Ready now:** [#53](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/53)

**Do not start:** [#57](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/57) (video) until #54–#56 done.

---

## ⌨️ Commands cheat sheet

| Command | When |
|---------|------|
| `/ship-issue #53` | Implement one issue end-to-end |
| `/ship-milestone M7.8` | Run or plan a phase / delivery train |
| `/verify` | Before every PR |

---

## 🧾 Human decisions log

| Decision | Default |
|----------|---------|
| Demo / video LLM | Anthropic Haiku via `LLM_PROVIDER=anthropic` |
| Self-host / air-gap | Ollama on Compose |
| First provider in code | `LLMProvider` protocol → Ollama + Anthropic + dummy |
| OpenAI | Optional after Anthropic works |
| Pitch vertical | Confidential **documents**, not legal-only |
| Support MVP / n8n CRM bot | Separate later project |
| Commit / merge | Train mode in AGENTS.md (orchestrator merges when green) |

---

## ✨ Success check (after Phase 1 + thin M8)

“I built private document Q&A with cited answers. The demo uses a fast API model; clients can run local Ollama on the same Docker stack. There’s a live pilot and a walkthrough video. When needed I can expose `/health` and `/chat`. I’d deploy this for a team with sensitive PDFs.”

If that’s true, you’re ready for client conversations with proof, not promises.
