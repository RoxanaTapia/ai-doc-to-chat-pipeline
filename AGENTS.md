# Contributing

This repository is private PDF Q&A. The layout is small on purpose so a change is easy to find and review.

## Tests

From the repo root:

```bash
pip install -r requirements.txt
pip install pytest ruff
PYTHONPATH=src pytest tests/ -q
ruff check .
```

CI runs the same pytest suite on every pull request.

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

## Review

Changes are reviewed and tested before merge. Keep secrets and real hostnames out of git.
