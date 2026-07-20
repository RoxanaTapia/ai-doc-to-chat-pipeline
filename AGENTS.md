# Agent orchestration playbook

## Background

How this repo uses Cursor **rules**, **subagents**, and **slash commands** to deliver milestones with clean one-issue-one-PR history.

**Operator guide:** [docs/operators/PROJECT-DIRECTION.md](docs/operators/PROJECT-DIRECTION.md) · **Roadmap:** [docs/operators/ROADMAP.md](docs/operators/ROADMAP.md) · **Docs index:** [docs/README.md](docs/README.md)

> **Takeaway:** One issue, one PR. Specialists edit; the orchestrator commits, merges, and keeps you in the loop with short pulses.

---

## 🛠️ How we build

This repo is set up for **Cursor agents**: named specialists under [`.cursor/agents/`](.cursor/agents/), slash commands under [`.cursor/commands/`](.cursor/commands/), and always-on rules under [`.cursor/rules/`](.cursor/rules/). See [`.cursor/README.md`](.cursor/README.md) for a one-screen map.

**Discipline stays simple:**

1. One GitHub issue → one branch `feat/<phase>-<name>` → one PR → `Closes #NN`.
2. Specialists edit only what they own; they do **not** commit.
3. The **milestone-orchestrator** commits, opens the PR, and (in train mode) merges when verifier + CI are green.
4. After each merge, a short **status pulse**; hard gates get a **Blocker** card instead of guesswork.

Operator direction and sequencing live in [docs/operators/PROJECT-DIRECTION.md](docs/operators/PROJECT-DIRECTION.md) and [docs/operators/ROADMAP.md](docs/operators/ROADMAP.md).

---

## 🗺️ Milestone recap

| Milestone | Goal | Status |
|-----------|------|--------|
| **M7** | Reference deployment (Docker, Compose, Caddy, live pilot) | ✅ Shipped |
| **M7.8** | Demo-ready tier: swappable LLM, Anthropic path, streaming | ✅ Shipped (#53–#56; #57 video open, last) |
| **M7.9** | Interface polish | ✅ Shipped (#70–#73) |
| **M7.95** | Sources trust | ✅ Shipped (#80–#83) |
| **M7.96** | Repo clarity (deploy/ consolidation, no shims) | ✅ Shipped (#89–#93) · **main only** |
| **Packaging** | Calm product framing (pilot + Cloud links; video URL later) | Soft pass after M7.96 |
| **M8** | Thin FastAPI `/health`, `/chat`, OpenAPI | Next (#58–#60) |
| **Video (#57)** | Walkthrough linked from README | **Last** after thin M8 |
| **M8.5** | Eval report export | Optional (#61) |
| **M9–M11** | Persist, access control, ops runbook | Client-triggered |
| **M12** | Light services / tiers one-pager | Light |

**North star:** private document Q&A with cited answers. Support MVP / n8n CRM is a **separate** later project.

Full detail: [docs/operators/ROADMAP.md](docs/operators/ROADMAP.md).

---

## 🎯 Current focus

- **Ship next:** demo video (#57) last — thin M8 (#58–#60) shipped; need public walkthrough URL.
- **Do not** bump `deploy/stable` unless the operator asks (VPS/Cloud stay on v0.8.0).
- **Then pause** this repo for the Support MVP sibling unless a paid engagement needs more depth here.
- **Later / on demand:** M8.5, M9–M11.

### Delivery train (default queue)

```text
M7.96 ✅ → packaging (minus video) → #58 → #59 → #60 → #57 → [pause]
```

| Wave | Work | Agents | Human? |
|------|------|--------|--------|
| — | M7.8–M7.96 | shipped | — |
| 1 | **Packaging** (README framing, tiers, thumbnail; skip video URL) | docs-writer | Soft |
| 2–4 | **#58 → #59 → #60** thin M8 | rag-core → config → streamlit → verifier | Rare |
| 5 | **#57** demo video + README link | docs-writer prepares; **human records** | Closing gate: video URL |
| — | **Pause** | orchestrator “phase complete” pulse | Support MVP elsewhere |

---

## 👥 Agent roster

| Role (`name`) | Milestones | Owns | Must NOT touch |
|---------------|------------|------|----------------|
| `milestone-orchestrator` | All | Queue, branches, commits, PRs, **merges**, pulses | Direct app code edits |
| `deploy-engineer` | M7, M7.96, M11 | `deploy/**` (Dockerfile, Compose, Caddy, scripts) | `src/app.py`, `src/rag.py` |
| `config-guardian` | M7–M12 | `configs/**`, `.env.example` | Application logic |
| `rag-core-engineer` | M7.8, M7.95, M8, M9, M12 | `src/rag.py`, `src/sectioning.py`, `src/retrieval_quality.py`, `src/rag/**`, `src/api/**` | Streamlit layout polish, Docker |
| `streamlit-engineer` | Feature UI wiring | `src/app.py` session/chat/upload/Sources payload wiring | Docker, FastAPI internals, visual redesigns |
| `streamlit-ux-designer` | M7.9 UI polish | Layout, IA, microcopy, chat/sources readability | Docker, FastAPI, RAG providers |
| `docs-writer` | M7, M7.8, M10–M12 | `docs/**`, `DEPLOYMENT*.md`, **README**, PR/issue prose, **blocker card polish** | Python except docstrings |
| `verifier` | All | Runs pytest/ruff; `tests/**` fixes only | Feature implementation |
| `blocker-reporter` | All | Blocker summaries (structured) | Code changes |

Invoke by role name. Files live in `.cursor/agents/`.

**Train roster:** orchestrator always on; **docs-writer on packaging**; rag-core + config + streamlit on thin M8 (#58–#60); docs-writer on #57 (video last) + pulses.

---

## 🗺️ Issue → agent mapping

### M7 ✅ shipped (#33–#39)

| Issue | Primary | Status |
|-------|---------|--------|
| #33–#39 | see git history | Done. Live at ai-doc-pilot.roxanatapia.dev |

### M7.8: Demo-ready tier (ship first)

| Issue | Primary | Secondary | Est. commits | Serial |
|-------|---------|-----------|--------------|--------|
| #53 M7.8-1 LLMProvider + env | rag-core-engineer | config-guardian | 2–3 | - |
| #54 M7.8-2 Anthropic adapter | rag-core-engineer | config-guardian | 2 | after #53 |
| #55 M7.8-3 Streamlit streaming | streamlit-engineer | rag-core-engineer (review) | 1–2 | after #54 |
| #56 M7.8-4 docs + sample doc | docs-writer | - | 1–2 | parallel #55 |
| #57 M7.8-5 demo video + README | docs-writer | human records | 1 | **last** after #58–#60 |

### M7.9: Interface polish (short UX pass)

| Issue | Primary | Secondary | Est. commits | Serial |
|-------|---------|-----------|--------------|--------|
| #70 M7.9-1 visual foundation | streamlit-ux-designer | - | 1–2 | - |
| #71 M7.9-2 copy + empty/ready states | streamlit-ux-designer | docs-writer (tone) | 1 | after #70 |
| #72 M7.9-3 sidebar IA | streamlit-ux-designer | - | 1 | after #71 |
| #73 M7.9-4 chat + Sources readability | streamlit-ux-designer | streamlit-engineer (edge) | 1–2 | after #72 |

### M7.95: Sources trust (before video)

| Issue | Primary | Secondary | Est. commits | Serial |
|-------|---------|-----------|--------------|--------|
| #80 M7.95-1 Sources display cap | config-guardian | streamlit-engineer | 1 | ∥ #83 |
| #81 M7.95-2 sort on-section → score | rag-core-engineer | streamlit-engineer | 1–2 | after #80; prefer after #83 |
| #82 M7.95-3 answer-overlap filter | rag-core-engineer | streamlit-engineer | 2 | after #81 |
| #83 M7.95-4 tighter header chunking | rag-core-engineer | config-guardian | 1–2 | ∥ #80 |

### M7.96: Repo clarity (chore · main only · no shims)

| Issue | Primary | Secondary | Est. commits | Serial |
|-------|---------|-----------|--------------|--------|
| #89 M7.96-1 REPO-STRUCTURE | docs-writer | - | 1 | ∥ #91 |
| #90 M7.96-2 consolidate under `deploy/` | deploy-engineer | docs-writer | 2 | after #89 |
| #91 M7.96-3 agentic surface tidy | docs-writer | orchestrator | 1 | ∥ #89 |
| #92 M7.96-4 README + docs index | docs-writer | - | 1 | after #90 |
| #93 M7.96-5 templates + pre-commit | config-guardian | docs-writer | 1 | after #92 |

**Rule:** no root stub files (“Moved to deploy/…”). Update real paths. Do not ff `deploy/stable` in this milestone.

### M8: Thin FastAPI contract (after packaging; before video)

| Issue | Primary | Secondary | Est. commits | Serial |
|-------|---------|-----------|--------------|--------|
| #58 M8-1 extract `src/rag/` | rag-core-engineer | verifier | 2–4 | - |
| #59 M8-2 FastAPI `/health` `/chat` | rag-core-engineer | config-guardian | 2–3 | after #58 |
| #60 M8-3 Streamlit → API | streamlit-engineer | rag-core-engineer | 2 | after #59 |

### M8.5: Eval export (optional / next)

| Issue | Primary | Secondary | Est. commits |
|-------|---------|-----------|--------------|
| #61 M8.5-1 eval report export | rag-core-engineer | verifier | 2 |

---

## 🔀 Parallel vs serial

**Safe in parallel:** #56 docs-writer while #55 streamlit (different files).

**Must be serial:** #53 → #54 → #55 (`src/rag.py`, `src/app.py`); #58 → #59 → #60; verifier last before PR.

**Do not parallelize** two agents on `src/rag.py`, `src/app.py`, or `README.md` in one issue.

Only **milestone-orchestrator** runs `git commit`, push, and merge.

---

## 🚂 Train mode (low-touch orchestration)

Default for the delivery train: one conductor chat can run the full queue. Still **one issue = one branch = one PR**. Learning pass is optional and does **not** gate the next issue.

```mermaid
flowchart TB
  subgraph pulse [Operator supervision]
    S["Status pulse<br/>3–6 lines"]
    B["Blocker card<br/>only when stuck"]
  end

  O[milestone-orchestrator]

  subgraph ship [Per issue loop]
    R[Specialists]
    V[verifier]
    C[commit 1–2]
    P[PR + merge]
  end

  O --> R --> V --> C --> P
  P --> S
  R -.->|needs human| B
  B --> docs-writer
```

### Per-issue loop

1. Mark issue **In progress**; branch `feat/m7-8-<name>` or `feat/m8-<name>` from latest `main`.
2. Dispatch specialists by ownership (no same-file parallel).
3. Run **verifier** (`pytest`; docker build only if Dockerfile/compose changed).
4. Orchestrator commits (1–2 granular), pushes, opens PR with **`## Main contribution`** first, `Closes #NN`.
5. When CI is green and the issue checklist is met, **orchestrator merges**, then emits a **status pulse** and starts the next wave.

### Commit / push / merge policy (train mode)

| Action | Who | When |
|--------|-----|------|
| Edit code/docs | Specialists | Per issue ownership |
| `git commit` | Orchestrator only | After verifier green |
| Push + open PR | Orchestrator | After commits |
| Merge | Orchestrator | CI green + DoD checklist met |
| Learning pass | Human (optional, async) | Does not block next issue |

If the operator says **hold merges** or **propose only**, fall back to draft PR + wait for explicit `commit` / `push` / `merge`.

### Status pulse (after every merge or wave)

Keep it short. Operator supervises via these, not via “say commit.”

```markdown
## Pulse · #NN merged
- Done: <one outcome line>
- PR: #<pr> → closes #NN
- Next: <next issue or parallel pair>
- Need from you: nothing | see Blocker
```

### Hard human gates (stop the train)

1. **Secrets** — API keys and real hostnames stay in `.env` / VPS only; never in git.
2. **CI still red** after one focused fix attempt — emit a Blocker card; do not loop forever.
3. **#57 recording** — only when the train reaches #57 (after thin M8). Agents prepare script/README; only the human can film/upload and supply the public URL. Do **not** stop packaging or M8 waiting for a video link.

Soft gate: packaging thumbnail / README emphasis (default: calm product framing; human can tweak later).

---

## 🐙 GitHub workflow (single issue)

```mermaid
flowchart LR
  A[Pick issue] --> B[Conductor chat]
  B --> C["/ship-issue #NN"]
  C --> D[Specialists edit]
  D --> E[Verifier]
  E --> F[Orchestrator commits]
  F --> G[PR + CI]
  G --> H[Orchestrator merges]
  H --> I[Status pulse]
```

1. Pick issue → **In progress** on Project board.
2. Conductor chat → `/ship-issue #NN` (or continue train chat).
3. Branch: `feat/m7-8-<short-name>` (or `feat/m8-<short-name>`).
4. Specialists edit; orchestrator splits **1–2 granular commits**.
5. `verifier` before PR.
6. PR: **`## Main contribution`** first, `Closes #NN`. Push and merge when green (train mode).
7. Optional learning pass later (see PROJECT-DIRECTION).

For the full queue, prefer `/ship-milestone M8` (packaging → #58–#60 → #57) rather than a fresh chat per issue (unless context is huge).

---

## ⌨️ Slash commands

| Command | Purpose |
|---------|---------|
| `/ship-issue` | Ship one GitHub issue end-to-end |
| `/ship-milestone` | Run or plan a milestone train (M7.8–M12) |
| `/verify` | pytest + ruff; report gaps |

Command files live in [`.cursor/commands/`](.cursor/commands/). Stale M7-only aliases were removed (use `/ship-issue`).

---

## 🧾 Human decisions log

| Decision | Current default | Notes |
|----------|-----------------|-------|
| VPS provider | Hetzner CPX32 (~€15/mo) | Falkenstein |
| Demo / video LLM | **Anthropic Haiku** (`LLM_PROVIDER=anthropic`) | Fast recording; API key in `.env` only |
| Self-host / air-gap LLM | Ollama (`phi3:mini` CPU; `llama3.1:8b` if RAM allows) | Not for walkthrough video |
| Domain / HTTPS | ai-doc-pilot.roxanatapia.dev | M7-6 done |
| Pitch vertical | **Confidential documents** (not legal-only) | NDA = eval corpus |
| Commit / merge policy | **Train mode:** orchestrator commits, pushes, merges when verifier + CI green | Operator may say `hold merges` for propose-only |
| Private deploy repo | Deferred until M10 or first client | App stays public |
| OpenAI provider | After Anthropic (#54) | Optional third backend |
| Support MVP / n8n CRM bot | Separate later project | Not this repo’s M8+ north star |
| M9–M11 depth | Client-triggered | Not required for demo-ready pause |
| Thin M8 in delivery train | **Include** (#58–#60 after packaging) | Then #57 video last; pause for Support MVP |
| Demo video order | **Last** after thin M8 (#57) | Not a blocker for packaging or M8 |

---

## 🚧 Blocker template

When stuck, invoke **blocker-reporter**, then ask **docs-writer** to polish into a clear “need you” card. **STOP** the train until the human replies (or the default timeout applies).

```markdown
## Blocker · need you
- **Issue:** #NN
- **What I need:** (concrete ask, e.g. YouTube/Loom URL)
- **Why:** (what cannot ship without it)
- **What is ready:** (what agents already finished)
- **Options:** A / B with tradeoff (optional)
- **Default if no reply:** (from Human decisions log; include quiet period if useful)
- **Blocks:** list of files/issues
- **Reply with:** (exact shape of answer you need)
```

---

## 📚 Documentation maintenance

| File | Owner | Update when |
|------|-------|-------------|
| **README.md** | `docs-writer` | Phase changes; video link (client-facing) |
| **DEPLOYMENT.md** | `docs-writer` + `deploy-engineer` | LLM provider setup, Compose |
| **docs/operators/ROADMAP.md** | `docs-writer` | Milestone scope changes |
| **docs/operators/PROJECT-DIRECTION.md** | Human + orchestrator | Phase order, operator habits |
| **docs/README.md** | `docs-writer` | Docs structure / index |
| **PR / issue prose** | `docs-writer` | Short warm Main contribution; issue Outcome + DoD |
| **Blocker cards** | `blocker-reporter` + `docs-writer` polish | Human gates |
| **AGENTS.md** | Human + orchestrator | Train policy, issues, decision log |

Tone for operator docs and GitHub issues: calm and confident. Avoid “hire-me,” “Upwork niche,” or salesy framing. Client contact links in README are fine; milestone prose is not a pitch deck.

After each phase slice: dispatch `docs-writer` to sync README + ROADMAP. Before opening a PR, docs-writer may polish the PR body (and issue text when needed).

---

## 📋 Operator prompts

### Full delivery train (preferred)

```text
Act as milestone-orchestrator. Run the delivery train:
packaging (minus video) → #58 → #59 → #60 → #57.

- Follow docs/operators/PROJECT-DIRECTION.md, docs/operators/ROADMAP.md, and AGENTS.md
- Specialists must NOT commit; you commit, push, and merge when verifier + CI are green
- After each merge, send a short Status pulse
- On hard gates (secrets, CI still red): Blocker card (docs-writer polish) and STOP
- #57 video is last: stop only when the train reaches recording (need public URL)
- Do not start Support MVP in this repo; end with a phase-complete pulse
```

### Single issue (propose-only / hold merges)

```text
Act as milestone-orchestrator. Ship GitHub issue #NN. Hold merges.

- New branch from main: feat/<phase>-<short-name>
- Follow docs/operators/PROJECT-DIRECTION.md and docs/operators/ROADMAP.md
- Specialists must NOT commit; you split 1–2 granular commits
- Run verifier before PR
- Draft PR with Main contribution; wait for my commit / push / merge
- If blocker: invoke blocker-reporter + docs-writer polish and STOP
```
