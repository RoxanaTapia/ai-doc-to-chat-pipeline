# Repository structure

## Background

Operator addendum. The friendly map is [docs/README.md](../README.md). This page only records layout rules that are easy to break.

> **Takeaway:** Product in `src/` and `configs/`. App containers in `deploy/`. HTTPS and invites are not this repo.

---

## Deploy assets

The image and Compose files live under [`deploy/`](../../deploy/). `.dockerignore` stays at the repo root because the build context is the repo root.

```text
deploy/
├── Dockerfile
├── docker-compose.yml              app + ollama (+ api profile)
└── docker-compose.shared-edge.yml  Live VPS: alias `app` on network `edge`
```

- No root copies of Dockerfile or Compose files.
- No stub files that say “Moved to …”.
- No Caddyfile or invite service here. Canonical HTTPS: [roxanatapia-edge](https://github.com/RoxanaTapia/roxanatapia-edge).
- Local and VPS commands: [running.md](running.md).

---

## Docs split

| Folder | Role |
|--------|------|
| [`docs/README.md`](../README.md) | Map of the repo and how an answer is produced |
| [`docs/product/`](../product/) | Architecture, walkthrough, sample documents |
| [`docs/operators/`](./) | Contributor sequencing and how-tos |
| [`docs/archive/`](../archive/) | Historical eval notes |

---

## Cursor surface

Keep these in the open tree. They are how issues ship.

| Path | Role |
|------|------|
| [`AGENTS.md`](../../AGENTS.md) | One issue, one branch, one PR |
| [`.cursor/agents/`](../../.cursor/agents/) | Specialist prompts |
| [`.cursor/commands/`](../../.cursor/commands/) | `/ship-issue`, `/verify` |
| [`.cursor/rules/`](../../.cursor/rules/) | Always-on conventions |
