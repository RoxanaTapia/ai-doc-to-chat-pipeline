# Deployment Guide

Complete walkthrough for a **single-VM Ollama pilot**: Docker Compose on a VPS (production-shaped) or on your laptop (local dev). No source-code reading required.

Architecture overview: [docs/architecture-pilot.md](docs/architecture-pilot.md)

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Docker 24+** | With BuildKit enabled (`DOCKER_BUILDKIT=1`) |
| **Docker Compose v2** | `docker compose` (plugin), not legacy `docker-compose` |
| **git** | To clone the repository |
| **~4 GB free disk** | App image (PyTorch + embedding models) plus Ollama model weights in a named volume |
| **SSH access** | For VPS deployment |
| **Outbound internet** | First run downloads Hugging Face embedding weights and Ollama model blobs |

Optional: copy [`.env.example`](.env.example) to `.env` if you need to override generation settings (see [Environment variables](#environment-variables)).

---

## VPS sizing

Choose hardware to match the Ollama model you plan to run. The Compose stack defaults to **`phi3:mini`** on the app service — suitable for CPU-only demos.

| Profile | Example sizing | Model | Typical use | Cost band (placeholder) |
|---------|----------------|-------|-------------|-------------------------|
| **CPU demo** | 4 vCPU, 8 GB RAM (e.g. Hetzner **CX32** or similar) | `phi3:mini` | Pilot evaluations, small teams, low concurrency | ~€15/mo |
| **GPU** | 8 GB+ VRAM (dedicated GPU VPS or cloud GPU instance) | `llama3.1:8b` | Higher answer quality, faster inference | ~$50–$200+/mo (provider-dependent) |

**CPU vs GPU notes**

- **`phi3:mini`** — default for CPU-only hosts; lower memory footprint; first answers may take 30–60+ seconds on cold start.
- **`llama3.1:8b`** — better quality; practical on GPU or very large RAM hosts; on CPU-only VMs you may hit slow responses or OOM — prefer `phi3:mini` or reduce `OLLAMA_NUM_CTX` (see [Troubleshooting](#troubleshooting)).

Model weights persist in the Docker volume `ollama_models` across restarts; pull each model once per volume.

---

## Deploy on a VPS

Replace `YOUR_VPS_IP` with your server’s public address. HTTPS and access control ship in a follow-up update; until then, restrict port **8501** to trusted IPs.

### 1. Prepare the server

```bash
# Example: Ubuntu/Debian — install Docker (official docs may vary by distro)
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
# Install Docker Engine 24+ and Compose plugin per https://docs.docker.com/engine/install/

sudo usermod -aG docker "$USER"
# Log out and back in so group membership applies
```

**Firewall (interim, before HTTPS):** allow SSH and Streamlit from trusted sources only. Ollama stays on the internal Compose network — do **not** expose port 11434 publicly.

```bash
# Example (ufw) — adjust CIDR to your office/VPN
sudo ufw allow OpenSSH
sudo ufw allow from YOUR_TRUSTED_CIDR to any port 8501
sudo ufw enable
```

### 2. Clone the repository

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
```

### 3. Start Ollama and wait until healthy

The app service does not start until Ollama passes its health check (`ollama list` inside the container).

```bash
docker compose up -d ollama
docker compose ps ollama   # STATUS should show "healthy" (may take up to ~60s on first boot)
```

If status stays `starting`, inspect logs: `docker compose logs -f ollama`.

### 4. Pull a model (one time per volume)

```bash
# CPU demo default (matches docker-compose.yml OLLAMA_MODEL)
docker compose exec ollama ollama pull phi3:mini

# GPU / higher-quality hosts
# docker compose exec ollama ollama pull llama3.1:8b
```

If you pull `llama3.1:8b`, set `OLLAMA_MODEL=llama3.1:8b` via `.env` and `env_file` on the app service, or export before `docker compose up`.

### 5. Build and start the full stack

```bash
docker compose up --build -d
docker compose ps
```

Both services should be **Up**; `ollama` should remain **healthy**. Streamlit listens on **8501** on the host.

### 6. Open the app

From a machine that can reach the VPS:

```text
http://YOUR_VPS_IP:8501
```

Upload a PDF and ask a question. The first query may be slow while embedding weights download inside the app container.

---

## Verify deployment

Run these on the VPS from the project directory.

### Compose health

```bash
docker compose ps
```

Expect `ollama` **healthy** and `app` **Up**. The app waits on `depends_on: condition: service_healthy` so Streamlit does not race a still-booting Ollama daemon.

### Ollama CLI (models in volume)

```bash
docker compose exec ollama ollama list
```

You should see `phi3:mini` (or the model you pulled).

### Ollama HTTP API (`/api/tags`)

Ollama is **not** published to the host in the default `docker-compose.yml` (internal network only). Use a one-off `curl` container on the Compose network:

```bash
# Network name is usually <project-directory>_default
PROJECT=$(basename "$PWD")
docker run --rm --network "${PROJECT}_default" curlimages/curl:8.5.0 \
  -s http://ollama:11434/api/tags
```

You should get JSON listing installed models. If the network name differs:

```bash
docker network ls | grep default
# Then: docker run --rm --network YOUR_NETWORK_NAME curlimages/curl:8.5.0 -s http://ollama:11434/api/tags
```

**Optional — host `curl`:** temporarily add `ports: ["11434:11434"]` under the `ollama` service locally (do not commit if you prefer internal-only), then:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Remove the port mapping when finished.

### Streamlit UI

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501/_stcore/health
```

Expect `200`. Then open `http://YOUR_VPS_IP:8501` in a browser and confirm chat works after a PDF upload.

---

## Local development

Same Compose flow on your laptop — useful before promoting to a VPS.

### Build image only (smoke test, no Ollama)

```bash
docker build -t ai-doc-to-chat .
docker run --rm -p 8501:8501 ai-doc-to-chat
```

Open [http://localhost:8501](http://localhost:8501). For real Ollama answers, use Docker Compose below (or run Ollama on the host and set `OLLAMA_HOST`).

### Docker Compose (app + Ollama)

#### Fresh start (end-to-end)

1. Start Ollama and wait until healthy:

   ```bash
   docker compose up -d ollama
   docker compose ps ollama   # STATUS should show "healthy"
   ```

2. Pull a model **once** into the persistent `ollama_models` volume:

   ```bash
   docker compose exec ollama ollama pull phi3:mini
   # docker compose exec ollama ollama pull llama3.1:8b   # GPU hosts
   ```

3. Start the app (foreground or detached):

   ```bash
   docker compose up --build
   # or: docker compose up --build -d
   ```

4. Optional: `cp .env.example .env`, edit values, add `env_file: .env` under the `app` service in `docker-compose.yml`.

5. Open [http://localhost:8501](http://localhost:8501), upload a PDF, and ask a question.

The app receives `OLLAMA_HOST=http://ollama:11434`, `USE_DUMMY_GENERATOR=false`, and `OLLAMA_MODEL=phi3:mini` from Compose.

#### One-shot equivalent

After models are already pulled:

```bash
docker compose up --build
```

Model weights persist in the `ollama_models` volume across restarts.

### Environment variables

Pilot-relevant settings are documented in [`.env.example`](.env.example).

| Variable | Compose default | Purpose |
|----------|-----------------|--------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API URL (Compose service name, not `localhost`) |
| `USE_DUMMY_GENERATOR` | `false` | Must be `false` for real Ollama answers on Compose/VPS |
| `OLLAMA_MODEL` | `phi3:mini` | Override; YAML default in `configs/config.yaml` is `llama3.1:8b` |

To override via file: `cp .env.example .env`, edit values, and add `env_file: .env` under the `app` service. Generation tuning (`temperature`, `num_ctx`, etc.) follows `configs/config.yaml` → `rag.generation` unless set as `OLLAMA_*` in `.env`.

See also [docs/architecture-pilot.md](docs/architecture-pilot.md#environment-variables-pilot).

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|--------|----------------|------------|
| `app` never starts / stays "Waiting" | Ollama not healthy yet | `docker compose logs ollama`; wait for `healthy` in `docker compose ps`. Slow disks may need more time within the healthcheck `start_period`. |
| **Connection refused** to Ollama | Ollama down, wrong host, or app started before Ollama was ready | Use Compose (not `docker run` app alone). On VPS, app must use `OLLAMA_HOST=http://ollama:11434` — not `localhost`. Check `docker compose ps` shows `ollama` **healthy**. |
| **Connection refused** on `YOUR_VPS_IP:8501` | Firewall, wrong IP, or app not running | `docker compose ps`; open port 8501 only to trusted CIDRs; confirm `curl http://127.0.0.1:8501/_stcore/health` on the VPS returns 200. |
| **Model not found** / pull errors in chat | No model in the volume or name mismatch | `docker compose exec ollama ollama pull phi3:mini` (or `llama3.1:8b`). Set `OLLAMA_MODEL` to the same tag. Verify with `docker compose exec ollama ollama list`. |
| **`/api/tags` empty or curl fails** | Model not pulled or wrong Docker network | Pull the model again; use the `docker run ... curlimages/curl` command under [Verify deployment → Ollama HTTP API](#verify-deployment) with the correct network name. |
| Slow first answer | Cold model + embedding download | Normal once per deploy; subsequent questions faster. |
| **OOM** / container killed / very slow on CPU | Model too large for RAM | Use `phi3:mini`; lower `OLLAMA_NUM_CTX` in `.env`; size up the VPS or move to a GPU instance with `llama3.1:8b`. Check `docker compose logs app` and `docker compose logs ollama`. |
| Disk full during pull | Undersized volume | Ensure ~4 GB+ free for app image and models; `docker system df`; prune unused images only if safe. |

Quick checks:

```bash
docker compose ps
docker compose logs --tail=50 ollama
docker compose logs --tail=50 app
docker compose exec ollama ollama list
```

---

## Notes

- No secrets are baked into the image; use runtime env vars or a mounted `.env` file (never commit `.env`).
- First query may be slow while Hugging Face embedding weights download into the app container cache.
- HTTPS, basic auth, and a public demo hostname — add in a follow-up deployment update; until then restrict port **8501** to trusted IPs (see [Deploy on a VPS](#deploy-on-a-vps)).
