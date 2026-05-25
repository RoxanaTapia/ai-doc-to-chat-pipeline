# Agent orchestration playbook

How this repo uses Cursor **rules**, **subagents**, and **slash commands** to deliver milestones M7–M12 with clean one-issue-one-PR history.

Full roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) · M7 GitHub milestone: [#1](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/milestone/1)

---

## Agent roster

| Role (`name`) | Milestones | Owns | Must NOT touch |
|---------------|------------|------|----------------|
| `milestone-orchestrator` | All | Reads issues, branches, PR plan | Direct code edits |
| `deploy-engineer` | M7, M11 | `Dockerfile`, `docker-compose*.yml`, `Caddyfile`, `deploy/` | `src/app.py`, `src/rag.py` |
| `config-guardian` | M7–M12 | `configs/**`, `.env.example` | Application logic |
| `rag-core-engineer` | M8, M9, M12 | `src/rag.py`, `src/rag/**`, `src/api/**` | Streamlit UI, Docker |
| `streamlit-engineer` | All UI | `src/app.py` | Docker, FastAPI internals |
| `docs-writer` | M7, M10–M12 | `docs/**`, `DEPLOYMENT*.md`, **README** (status, deploy pointers, production vision — not app code) | Python except docstrings |
| `verifier` | All | Runs pytest/ruff; `tests/**` fixes only | Feature implementation |
| `blocker-reporter` | All | Blocker summaries | Code changes |

Invoke by role name: `deploy-engineer`, not persona names. Files live in `.cursor/agents/`.

---

## M7 issue → agent mapping

| Issue | Primary | Secondary (parallel) | Est. commits |
|-------|---------|----------------------|--------------|
| #33 M7-1 Dockerfile | deploy-engineer | — | 1–2 |
| #34 M7-2 compose | deploy-engineer | config-guardian | 1–2 |
| #35 M7-3 health gate | deploy-engineer | docs-writer | 1–2 |
| #36 M7-4 env | config-guardian | docs-writer | 1 |
| #37 M7-5 DEPLOYMENT.md | docs-writer | deploy-engineer (review) | 1–2 |
| #38 M7-6 Caddy | deploy-engineer | — | 2–3 | **Done** — live at ai-doc-pilot.roxanatapia.dev |
| #39 M7-7 demo assets | docs-writer | — | 1–2 |

---

## Parallel vs serial

**Safe in parallel:** M7-1 Dockerfile + M7-4 `.env.example`; docs-writer + config-guardian on different files.

**Must be serial:** M7-2 after M7-1 merges; same-file edits (`docker-compose.yml`, `src/app.py`); verifier last before PR.

**Do not parallelize** two agents on `docker-compose.yml`, `src/app.py`, or `README.md` in one issue.

Only **milestone-orchestrator** runs `git commit` after parallel subagents return.

---

## GitHub workflow

1. Pick issue → move to **In progress** on Project board.
2. Run `/ship-m7-issue` or orchestrator prompt with issue `#NN`.
3. Branch: `feat/m7-<short-name>` (one branch per issue).
4. Specialists edit; orchestrator splits **1–2 granular commits** (see ROADMAP).
5. `verifier` runs before PR.
6. PR body: **`## Main contribution`** paragraph first (outcome, why it matters), then Summary, Test plan, and `Closes #NN`. Push/merge only when human approves.

---

## Slash commands

| Command | Purpose |
|---------|---------|
| `/ship-m7-issue` | Ship one M7 GitHub issue end-to-end |
| `/ship-milestone` | Plan or batch a milestone (M7–M12) |
| `/verify` | pytest + ruff; report gaps |
| `/parallel-m7-bootstrap` | One-time parallel M7-1 + M7-4 + DEPLOYMENT skeleton |

---

## Human decisions log

Defaults when orchestrator would otherwise stall. **Update this table when you decide.**

| Decision | Current default | Notes |
|----------|-----------------|-------|
| VPS provider | Hetzner CPX32 (~€15/mo) | **Running** — `bougie-main-01`, Falkenstein |
| Ollama model (demo CPU) | `phi3:mini` | `llama3.1:8b` when GPU available; `OLLAMA_NUM_CTX=1024` on 8 GB |
| Domain / HTTPS | `ai-doc-pilot.roxanatapia.dev` | **Live** — Let's Encrypt via Caddy, M7-6 done |
| Commit policy | Orchestrator proposes; human says `commit` | No push without explicit approval |
| Private deploy repo | Deferred until M10 or first client | App code stays public |
| LLM for pilots | Ollama default; Anthropic in M12 | Same RAG, swappable backend |

---

## Blocker template

Specialists and `blocker-reporter` use this format:

```markdown
## Blocker
- **Issue:** #NN or M7-x
- **Decision needed:** (e.g. VPS provider, domain, model size)
- **Options:** A / B with cost tradeoff
- **Default if no reply:** (from Human decisions log above)
- **Blocks:** list of files/issues
```

---

## Documentation maintenance

Keep user-facing docs aligned when milestones ship or deployment behavior changes.

| File | Owner | Update when |
|------|-------|-------------|
| **README.md** | `docs-writer` | Milestone status changes, new deploy paths — **client-facing only** |
| **DEPLOYMENT.md** | `docs-writer` + `deploy-engineer` review | Compose, env, VPS, HTTPS, troubleshooting (technical buyers + IT) |
| **docs/ROADMAP.md** | `docs-writer` | M7–M12 scope or definition-of-done changes (contributors) |
| **docs/architecture-pilot.md** | `docs-writer` | Architecture target changes (client/IT summary) |
| **docs/demo-script.md** | `docs-writer` | Public stub only — link placeholder until video ships |
| **`docs-private/`** | Human + `docs-writer` | **Local only (gitignored)** — sales playbook, env switches, full roadmap detail, demo recording script, infra notes |
| **AGENTS.md** | Human + orchestrator | New agents, decision log, issue→agent map |

**Audience split**

| Audience | Read | Do not put here |
|----------|------|-----------------|
| **Clients / buyers** | README | Issue numbers, agent names, env priority chains, sales scripts |
| **Client IT** | DEPLOYMENT.md, architecture-pilot | Internal pricing, VPS provider picks, funnel scripts |
| **You / operators** | `docs-private/` | Real hostnames, secrets, client names |
| **Contributors** | AGENTS.md, ROADMAP | Sales playbook |

**Bootstrap `docs-private/`** on a new machine: copy or recreate the folder locally (see index in your existing `docs-private/README.md`). It is never committed.

**After merging a GitHub milestone slice** (e.g. M7 complete, M8 started):

1. Orchestrator or human opens a **docs issue** or adds sub-task: “Sync README + ROADMAP status.”
2. Dispatch **`docs-writer` only** — do not parallelize with code issues on the same PR unless docs-only.
3. **README** stays **short and client-facing** — no operator jargon, env implementation detail, milestone issue numbers, or duplicate deploy commands. Link to `DEPLOYMENT.md` for setup ([`docs-commercial` rule](.cursor/rules/docs-commercial.mdc)).
4. **Operator / sales detail** (funnel scripts, VPS provider picks, “the switches”, clone-vs-URL hosting, demo recording checklist) lives in **`docs-private/`** on your machine — never committed. After shipping public docs, sync the matching `docs-private/` file if one exists.

There is **no separate README agent** — use **`docs-writer`** for README status, deploy sections, and production narrative.

---

- **Public repo:** app code, generic Compose, `DEPLOYMENT.md`, agents/rules (this file).
- **Private deploy repo (later):** client hostnames, SSO configs, production secrets — not duplicated app logic.

---

## Operator prompt (copy per issue)

```text
Act as milestone-orchestrator. Ship GitHub issue #NN.

- Branch: feat/m7-<short-name>
- Follow docs/ROADMAP.md and milestone-workflow rules
- Specialists must NOT commit; you split 1–2 granular commits
- Run verifier before PR
- If blocker: invoke blocker-reporter and STOP
- Do not push unless I say push
```
