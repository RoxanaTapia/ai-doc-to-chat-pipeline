# Repository structure

## Background

As of **M7.96-2** ([#90](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/90)), the repo reads as one calm tree: product code and docs in the obvious places, deploy assets under `deploy/`, and agentic tooling kept intentional rather than hidden. This page describes the **current** layout.

> **Takeaway:** Product in `src/` + `docs/product/`. Ops in `deploy/` + `DEPLOYMENT.md`. Agents in `AGENTS.md` + `.cursor/`. No root “Moved to deploy/…” stubs.

**Milestone rule:** M7.96 merges to **`main` only**. Do **not** bump `deploy/stable` for this chore (VPS and Streamlit Cloud stay on the current pin until you choose otherwise).

---

## 🗺️ Top-level layout

```text
.
├── README.md                 ← client-facing overview
├── AGENTS.md                 ← contributor / agent playbook
├── DEPLOYMENT.md             ← self-host & pilot ops
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .pre-commit-config.yaml
├── .streamlit/               ← Streamlit theme / config
├── .cursor/                  ← rules, agents, slash commands
├── .github/                  ← issue templates, CI
├── src/                      ← application code
├── tests/
├── configs/                  ← YAML (chunking, prompts, …)
├── docs/                     ← product / operators / archive
└── deploy/                   ← Docker, Compose, Caddy, scripts
```

| Area | Role |
|------|------|
| **Root prose** | README (buyers), DEPLOYMENT (IT), AGENTS (contributors) |
| **`src/`** | Streamlit app, RAG, retrieval helpers |
| **`tests/`** | pytest suite |
| **`configs/`** | Tunables loaded at runtime |
| **`docs/`** | Audience-split documentation (see below) |
| **`deploy/`** | Everything needed to run the pilot container stack |
| **`.cursor/` + `AGENTS.md`** | How humans and agents ship milestones |

Contributors may run `pre-commit install` so local commits use [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) (ruff + basic hooks).

---

## 🛠️ `deploy/` (no root shims)

Deploy assets live **only** under `deploy/` (as of M7.96-2 / #90). `.dockerignore` stays at the **repo root** (build context is the repo root).

```text
deploy/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.caddy.yml
├── Caddyfile
├── Caddyfile.ip
├── caddy-basicauth.conf.example
├── generate-caddy-auth.sh
├── generate-ip-tls.sh
└── …                             ← other deploy scripts as needed
```

**Canonical Compose (from repo root):**

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.caddy.yml up --build -d
```

**Rules**

- No root copies of Dockerfile, `docker-compose*.yml`, or `Caddyfile*`.
- No stub files at the root that say “Moved to `deploy/`…”. Update real paths in docs and CI instead.
- Compose build context is `..` (repo root); `dockerfile: deploy/Dockerfile` (path relative to context).

Operators still start from [DEPLOYMENT.md](../../DEPLOYMENT.md); the folder above is where the files actually live.

---

## 📖 `docs/` map

Unchanged by the Docker move; kept here so the full tree is one picture.

```text
docs/
├── README.md          ← docs index (start here)
├── product/           ← architecture, demo, evaluation, samples
├── operators/         ← roadmap, direction, this file, OCR testing
└── archive/           ← historical eval rounds
```

| Folder | Audience |
|--------|----------|
| [`docs/product/`](../product/) | Buyers and IT evaluating the product |
| [`docs/operators/`](./) | Contributors shipping the delivery train |
| [`docs/archive/`](../archive/) | Older eval notes; not day-one reading |

Index: [docs/README.md](../README.md).

---

## 🎯 Agentic surface (intentional)

Agentic files stay in the open tree on purpose:

| Path | Purpose |
|------|---------|
| [`AGENTS.md`](../../AGENTS.md) | Issue → agent map, train mode, human gates |
| [`.cursor/agents/`](../../.cursor/agents/) | Specialist role prompts |
| [`.cursor/rules/`](../../.cursor/rules/) | Always-on coding and milestone rules |
| [`.cursor/commands/`](../../.cursor/commands/) | Slash commands (`/ship-issue`, `/verify`, …) |

Tidy and naming hygiene for this surface is [#91](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/91). Do not bury or delete the agentic layer for “cleaner GitHub cosmetics.”

---

## ✅ What this milestone does not change

- **Pin / release:** `deploy/stable` is not fast-forwarded by M7.96.
- **Product features:** No RAG or UI work in this chore.
- **Docs audiences:** `product/` vs `operators/` vs `archive/` stay as they are; README polish is [#92](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues/92).

Related: [ROADMAP · M7.96](ROADMAP.md#m796-repo-clarity-chore--main-only).
