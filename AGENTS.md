# Project guide

## What we're building

This repository answers a question from a PDF and shows the best-matching page. The Streamlit demo in `apps/demo` is the pilot. The core in `src/ai_doc` reads the file, searches it, and writes the answer. Product and operator pages start at [docs/README.md](docs/README.md). How to run the app is in [docs/operators/running.md](docs/operators/running.md).

## How a change happens

1. **One issue.** The issue is the request. Use [.github/ISSUE_TEMPLATE.md](.github/ISSUE_TEMPLATE.md). The title starts with one emoji code, such as 🛠️ for code and tooling or 📄 for a document, then a short sentence.
2. **One branch.** The branch name is plain words, such as `agent-docs`. This work stays on that branch.
3. **One pull request.** The title is a plain sentence. The body follows [.github/pull_request_template.md](.github/pull_request_template.md). Roxana reviews and merges.
4. **Automated tests on every change.** GitHub Actions runs the commands in [Tests](#tests) on every pull request. Run them before the pull request is ready.

One issue, one branch, one pull request.

## Where things are

| Path | What it is |
|---|---|
| [README.md](README.md) | What the pilot is, written to you |
| [src/ai_doc/](src/ai_doc/) | Retrieval core and the optional FastAPI layer |
| [apps/demo/](apps/demo/) | Streamlit demo, the reference client |
| [configs/](configs/) | Chunking, retrieval, and prompts |
| [tests/](tests/) | Automated tests |
| [deploy/](deploy/) | App image and Compose |
| [docs/](docs/README.md) | Product and operator pages |
| [vendor/demo-kit/](vendor/demo-kit/) | Shared header, theme, icons, and fonts |
| [.env.example](.env.example) | Product knobs. Copy to `.env`. Never commit a filled file |
| [AGENTS.md](AGENTS.md) | This guide |
| [.cursor/rules/](.cursor/rules/) | How agents are allowed to work |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | Style check and automated tests |
| [.github/ISSUE_TEMPLATE.md](.github/ISSUE_TEMPLATE.md) | Short form for a new issue |
| [.github/pull_request_template.md](.github/pull_request_template.md) | Short form for a pull request |

## Tests

From the repo root, on Python 3.12:

```bash
pip install -r requirements.txt
pip install pytest "ruff==0.17.2"
ruff check .
ruff format --check .
sh vendor/demo-kit/check-hash.sh
PYTHONPATH=src:apps/demo pytest tests/ -q
```

`vendor/demo-kit/VERSION` must read `2`. CI installs Playwright and Chromium before pytest so the browser tests can run. A browser test skips when Playwright is missing.

## For the agents

Do:

- Read this guide and stay inside the current issue.
- Keep one issue, one branch, and one pull request.
- Run the tests above before the pull request is ready.
- Match the code that is already here. Formatting does not change what the app does.
- Keep secrets out of git. New product settings go in `.env.example` with an empty value.
- Write plain words, with no em dashes.

Don't:

- Merge. Roxana reviews and merges.
- Force-push, rebase a shared branch, or change who can see the repository.
- Weaken a test so it passes.
- Invent numbers, prices, or hosting details in public writing.
- Add a second Ruff config. The only one is in `pyproject.toml`.
