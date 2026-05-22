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

Open [http://localhost:8501](http://localhost:8501). For Ollama-backed answers, run Ollama on the host and pass `OLLAMA_HOST` (Compose wiring arrives in M7-2).

## Notes

- No secrets are baked into the image; use runtime env vars or a mounted `.env` file.
- First query may be slow while Hugging Face embedding weights download into the container cache.
