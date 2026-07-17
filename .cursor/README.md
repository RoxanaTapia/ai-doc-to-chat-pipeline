# Cursor workspace helpers

Short map of what lives under `.cursor/` for this repo. Full workflow lives in [`AGENTS.md`](../AGENTS.md).

| Folder | Role |
|--------|------|
| **`agents/`** | Specialist prompts (orchestrator, docs, deploy, RAG, Streamlit, verifier, …). Invoke by role name. |
| **`commands/`** | Slash commands: `/ship-issue`, `/ship-milestone`, `/verify`. |
| **`rules/`** | Always-on coding and milestone conventions applied in Cursor chats. |

> **Takeaway:** One GitHub issue → one branch → one PR. Specialists edit; the orchestrator commits and merges.
