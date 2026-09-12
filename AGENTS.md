# How this repo ships with agents

## Background

This product is private PDF Q&A. The code should stay easy to follow. The Cursor setup under [`.cursor/`](.cursor/) exists so changes stay small, owned, and reviewable, instead of one chat rewriting half the tree.

> **Takeaway:** One issue → one branch → one PR. Specialists edit. The orchestrator commits. You get a short status pulse, not a mystery merge.

Sequencing and roadmap live elsewhere: [PROJECT-DIRECTION.md](docs/operators/PROJECT-DIRECTION.md) · [ROADMAP.md](docs/operators/ROADMAP.md) · [docs map](docs/README.md).

---

## Why bother

| Without agents | With this setup |
|----------------|-----------------|
| One long chat owns every file | Named roles own clear paths |
| History is hard to explain | Each PR closes one outcome |
| “Who changed RAG?” | `rag-core-engineer` (and friends) |
| You babysit every commit | Train mode: verify → PR → merge when green |

You still decide product direction. Agents handle the mechanical loop.

---

## 🛠️ The rules that matter

1. **One GitHub issue** → branch `feat/<short-name>` → **one PR** → `Closes #NN`.
2. **Specialists edit only what they own.** They do not commit.
3. **`milestone-orchestrator` commits, opens the PR, and merges** when verifier + CI are green (unless you said `hold merges`).
4. **Do not parallelize** two agents on the same hot file (`src/rag/`, `src/app.py`, `README.md`).
5. **Secrets stay out of git.** API keys and real hostnames live in `.env` / the VPS only.

PR bodies start with **`## Main contribution`** (outcome first, not a file list).

---

## 👥 Who does what

| Role | Owns | Hands off |
|------|------|-----------|
| `milestone-orchestrator` | Queue, branches, commits, PRs, merges, pulses | App code edits |
| `rag-core-engineer` | `src/rag/`, `src/api/` | Streamlit polish, Docker |
| `streamlit-engineer` | `src/app.py` wiring (session, chat, Sources) | Docker, FastAPI internals |
| `streamlit-ux-designer` | Layout, IA, microcopy | RAG providers, Docker |
| `config-guardian` | `configs/`, `.env.example` | Application logic |
| `deploy-engineer` | `deploy/` (app image, Compose, shared-edge) | `src/app.py`, `src/rag/` |
| `docs-writer` | `docs/`, README, PR/issue prose | Python features |
| `verifier` | pytest; fixes only in `tests/` | Feature code |
| `blocker-reporter` | Structured “need you” cards | Code changes |

Prompts live in [`.cursor/agents/`](.cursor/agents/). Folder map: [`.cursor/README.md`](.cursor/README.md).

---

## ⌨️ How to run it

| Command | Use when |
|---------|----------|
| `/ship-issue #NN` | One issue, end to end |
| `/ship-milestone …` | A short queue of related issues |
| `/verify` | pytest (and docker build only if `deploy/` changed) |

**Single issue (hold merges):**

```text
Act as milestone-orchestrator. Ship GitHub issue #NN. Hold merges.
Specialists must NOT commit. Draft the PR; wait for my commit / push / merge.
```

**Train mode (default):** same, but the orchestrator pushes and merges when CI is green, then sends a short pulse.

```markdown
## Pulse · #NN merged
- Done: <one outcome line>
- PR: #<pr> → closes #NN
- Next: <next issue>
- Need from you: nothing | see Blocker
```

---

## 🚧 When to stop

Stop and ask with a **Blocker** card when:

- A secret or real hostname would have to land in git
- CI stays red after one focused fix
- Only a human can finish the step (for example a recording URL)

```markdown
## Blocker · need you
- **Issue:** #NN
- **What I need:** …
- **Why:** …
- **What is ready:** …
- **Reply with:** …
```

---

## 📚 Where detail lives

| Need | Open |
|------|------|
| Product story | [README.md](README.md) |
| What ships next | [ROADMAP.md](docs/operators/ROADMAP.md) |
| Operator habits | [PROJECT-DIRECTION.md](docs/operators/PROJECT-DIRECTION.md) |
| How to run the stack | [running.md](docs/operators/running.md) |
| Repo map | [docs/README.md](docs/README.md) |

Keep this file short. If a rule only matters for one milestone, put it in the roadmap, not here.
