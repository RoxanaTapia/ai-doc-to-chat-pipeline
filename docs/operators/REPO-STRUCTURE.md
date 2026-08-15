# Repository structure

## Background

Operator addendum. The friendly map is [docs/README.md](../README.md). This page only records layout rules that are easy to break.

> **Takeaway:** Product in `src/` and `configs/`. Ops in `deploy/`. Agents stay visible in `.cursor/` and [AGENTS.md](../../AGENTS.md).

---

## Deploy assets

Everything that runs the container stack lives under [`deploy/`](../../deploy/). `.dockerignore` stays at the repo root because the build context is the repo root.

```text
deploy/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.caddy.yml         Single-product HTTPS, or VPS rollback
├── docker-compose.shared-edge.yml   Portfolio VPS: alias `app` on network edge
├── Caddyfile
└── invite/                          Rollback copy; canonical is roxanatapia-edge
```

- No root copies of Dockerfile, Compose, or Caddy files.
- No stub files that say “Moved to `deploy/`…”.
- Dedicated VM: `docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up --build -d`
- Portfolio VPS: app uses `docker-compose.shared-edge.yml`; Caddy lives in [roxanatapia-edge](https://github.com/RoxanaTapia/roxanatapia-edge).

Install narrative: [DEPLOYMENT.md](../../DEPLOYMENT.md).

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
