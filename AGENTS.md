# Contributing

This repository is a public PDF Q&A project. The layout is small on purpose so a change is easy to find and review.

## Tests

From the repo root:

```bash
pip install -r requirements.txt
pip install pytest
PYTHONPATH=src pytest tests/ -q
```

CI runs this pytest suite on every pull request. `pyproject.toml` also has Ruff settings for local use.

## Layout

| Path | What it is |
|------|------------|
| `src/` | Streamlit app, retrieval, and the optional FastAPI layer |
| `configs/` | Chunking, retrieval, and prompt tunables |
| `tests/` | pytest suite |
| `deploy/` | App image and Compose (Ollama; optional API) |
| `docs/` | Product and operator pages. Start at [docs/README.md](docs/README.md) |
| `.env.example` | Product knobs. Copy to `.env` locally; never commit a filled file |

How to run the stack: [docs/operators/running.md](docs/operators/running.md).

## PRs and reviews

PR titles and descriptions are concise, consistent and easy to scan, so the PR history of the public repo tells a story.

### PR title

- Format: `type(scope): what changes for the user`, lowercase, imperative, at most about 60 characters, with no issue number in the title.
- Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`. Scope is the area, for example `ui`, `api`, `deploy`.
- Example: `fix(ui): keep header fixed on mobile`
- PRs are squash-merged, so the PR title becomes the single commit on main and the main history reads as one line per change.

### PR body

```markdown
## 🎯 What
One or two lines on the user-visible change. Closes #184

## 🧱 Commit story
- fix(ui): pin header height before fonts load
- test(ui): add header position test at 390px and 1440px

## 🧪 Test
- [ ] CI green
- [ ] up to 3 checks that prove the fix

## 🚀 Deploy
`docker compose -f deploy/docker-compose.yml up --build -d` · rebuilds `app` · no env or config changes · rollback: revert plus redeploy

<details><summary>Details</summary>screenshots, numbers, longer notes</details>
```

- **Commit story:** each commit in the PR is one logical step with the same `type(scope): ...` style. No "wip" or "address review" commits; clean them up before review.
- **Deploy:** name the exact command, the services rebuilt, any env or config changes, and the rollback. Docs-only changes write **No deploy**. Otherwise use a real command from [docs/operators/running.md](docs/operators/running.md).
- **Double-check every command:** each command in a PR or review (deploy command, path, flag, test command) must be checked against the repo or the docs before posting. Use only commands that appear in the repo, such as `docker compose -f deploy/docker-compose.yml up --build -d`, and never invented flags or paths. If a command could not be run or verified, say so in the PR.
- **Where this is written down:** `.github/pull_request_template.md` and this section. `.cursor/rules` points here.

### Review comment format

Every review comment is short, easy to scan, and uses emojis. It has a header line and at most five one-line bullets above a collapsed details block. This applies to code, deploy, live-test and copy reviews, and to the post-deploy OK.

```markdown
**✅ Pass · Deploy review · `abc1234`** (or ❌ Fail)
- ✅ what works
- ❌ blocking issue
- ⚠️ minor, non-blocking
- 🚀 deploy notes
<details><summary>Details</summary>numbers, logs, screenshots</details>
```

### Public repos

Client-facing: no prices, hostnames, hosting provider, server sizes, /root paths, private-repo links, internal notes, or "Cursor"/agent badges and links in PR bodies.
