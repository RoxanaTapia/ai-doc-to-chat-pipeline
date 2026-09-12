# Cursor workspace helpers

Short map of what lives under `.cursor/` for this repo. Why and how: [`AGENTS.md`](../AGENTS.md).

| Folder | Role |
|--------|------|
| **`agents/`** | Specialist prompts (orchestrator, docs, deploy, RAG, Streamlit, verifier, …). Invoke by role name. |
| **`commands/`** | Slash commands: `/ship-issue`, `/ship-milestone`, `/verify`. |
| **`rules/`** | Always-on coding conventions applied in Cursor chats. |

> **Takeaway:** One issue → one branch → one PR. Specialists edit; the orchestrator commits and merges.
