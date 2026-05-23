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

### Fresh start (end-to-end)

1. Start Ollama and wait until it is healthy (the app service does not start until then):

   ```bash
   docker compose up -d ollama
   docker compose ps ollama   # STATUS should show "healthy"
   ```

2. Pull a model **once** into the persistent `ollama_models` volume (choose one):

   ```bash
   # CPU demo default (see AGENTS.md)
   docker compose exec ollama ollama pull phi3:mini

   # GPU / more capable hosts
   docker compose exec ollama ollama pull llama3.1:8b
   ```

3. Start the app (or bring up the full stack):

   ```bash
   docker compose up --build
   ```

   Compose starts `ollama` first; `app` waits on `depends_on: condition: service_healthy` so Streamlit does not race a still-booting Ollama daemon.

4. Optional: match the pulled model in a local `.env` (e.g. `OLLAMA_MODEL=phi3:mini`), add `env_file: .env` under the `app` service, or export before `docker compose up`.

5. Open [http://localhost:8501](http://localhost:8501), upload a PDF, and ask a question. The first run may be slow while Hugging Face embedding weights download inside the app container.

The app service receives `OLLAMA_HOST=http://ollama:11434` and `USE_DUMMY_GENERATOR=false`.

### Environment variables

Pilot-relevant settings are documented in [`.env.example`](.env.example). For self-hosted deployments:

| Variable | Compose default | Purpose |
|----------|-----------------|--------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API URL (service name in Compose, not `localhost`) |
| `USE_DUMMY_GENERATOR` | `false` | Must be `false` for real Ollama answers on Compose/VPS |
| `OLLAMA_MODEL` | `phi3:mini` | Override; YAML default is `llama3.1:8b` in `configs/config.yaml` |

To override via a file: `cp .env.example .env`, edit values, and add `env_file: .env` under the `app` service in `docker-compose.yml`. Generation tuning (`temperature`, `num_ctx`, etc.) follows `configs/config.yaml` → `rag.generation` unless set as `OLLAMA_*` in `.env`.

See also [docs/architecture-pilot.md](docs/architecture-pilot.md#environment-variables-pilot).

### One-shot equivalent

If you prefer a single command after models are already pulled:

```bash
docker compose up --build
```

Model weights still persist in the `ollama_models` volume across restarts.

## Troubleshooting

| Symptom | Likely cause | What to do |
|--------|----------------|------------|
| `app` never starts / stays "Waiting" | Ollama not healthy yet | `docker compose logs ollama`; wait for `healthy` in `docker compose ps`. Increase `start_period` on slow disks if needed. |
| **Connection refused** / cannot connect to Ollama | Ollama down, wrong host, or app started before Ollama was ready | Use Compose (not `docker run` app alone) or set `OLLAMA_HOST` to a running server. Check `docker compose ps` shows `ollama` healthy. |
| **Model not found** / pull errors in chat | No model in the volume | Run `docker compose exec ollama ollama pull phi3:mini` (or `llama3.1:8b`). Set `OLLAMA_MODEL` to the same name. |
| Slow or OOM on first answer | Large default model on CPU | Use `phi3:mini` and/or lower `OLLAMA_NUM_CTX` in `.env`. |

Verify Ollama from the host (optional):

```bash
docker compose exec ollama ollama list
```

## Notes

- No secrets are baked into the image; use runtime env vars or a mounted `.env` file.
- First query may be slow while Hugging Face embedding weights download into the container cache.
