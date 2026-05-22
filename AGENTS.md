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
| `docs-writer` | M7, M10–M12 | `docs/**`, `DEPLOYMENT*.md`, README deploy sections | Python except docstrings |
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
| #38 M7-6 Caddy | deploy-engineer | blocker-reporter if no domain | 2–3 |
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
6. PR body: `Closes #NN`. Push/merge only when human approves.

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
| VPS provider | Hetzner CX32 (~€15/mo) | Defer purchase until M7-6 |
| Ollama model (demo CPU) | `phi3:mini` | `llama3.1:8b` when GPU available |
| Domain / HTTPS | Defer #38 until domain exists | Interim: IP + basic auth |
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

## Repository strategy

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
