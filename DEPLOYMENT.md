# Deployment Guide

> Stub for M7-1. Full Ollama, Compose, and VPS instructions land in M7-5.

## Prerequisites

- Docker 24+ with BuildKit enabled
- ~4 GB free disk for the image (PyTorch + embedding models download on first run)

## Build image

```bash
docker build -t ai-doc-to-chat .
```

## Run container (local smoke test)

```bash
docker run --rm -p 8501:8501 ai-doc-to-chat
```

Open [http://localhost:8501](http://localhost:8501). For Ollama-backed answers, use Docker Compose (below) or run Ollama on the host and pass `OLLAMA_HOST`.

## Docker Compose (app + Ollama)

Start both services on the default Compose network:

```bash
docker compose up --build
```

The app service receives `OLLAMA_HOST=http://ollama:11434` and `USE_DUMMY_GENERATOR=false`. Ollama model weights persist in the `ollama_models` volume.

Pull a model once (CPU demo default from [AGENTS.md](AGENTS.md)):

```bash
docker compose exec ollama ollama pull phi3:mini
```

Optional: set `OLLAMA_MODEL=phi3:mini` in a local `.env` file and add `env_file: .env` under the `app` service, or export it before `docker compose up`.

Open [http://localhost:8501](http://localhost:8501) and ask a question after uploading a PDF. First run may be slow while embedding weights download inside the app container.

## Notes

- No secrets are baked into the image; use runtime env vars or a mounted `.env` file.
- First query may be slow while Hugging Face embedding weights download into the container cache.
