# Repository structure

## Background

Operator addendum. The friendly map is [docs/README.md](../README.md). This page only records layout rules that are easy to break.

> **Takeaway:** Product in `src/` and `configs/`. App containers in `deploy/`. HTTPS belongs with the reverse proxy in front.

---

## Deploy assets

The image and Compose files live under [`deploy/`](../../deploy/). `.dockerignore` stays at the repo root because the build context is the repo root.

```text
deploy/
├── Dockerfile
├── docker-compose.yml              app + ollama (+ api profile)
└── docker-compose.shared-edge.yml  Behind a reverse proxy: alias `app`, no published port
```

- No root copies of Dockerfile or Compose files.
- No stub files that say “Moved to …”.
- No reverse proxy or invite service here. Public HTTPS belongs with an existing reverse proxy.
- Local and reverse-proxy commands: [running.md](running.md).

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

Keep these in the open tree. They are short and public.

| Path | Role |
|------|------|
| [`AGENTS.md`](../../AGENTS.md) | Tests, layout, and review |
| [`.cursor/rules/`](../../.cursor/rules/) | Coding conventions |
