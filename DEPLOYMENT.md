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

Choose hardware to match the Ollama model you plan to run. The Compose stack defaults to **`llama3.1:8b`** (override with `OLLAMA_MODEL=phi3:mini` in `.env` for CPU-only demos).

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

Replace `YOUR_VPS_IP` with your server’s public address, or use a subdomain with the [HTTPS (Caddy)](#https-and-basic-auth-caddy) overlay for the recommended pilot setup.

### What a client (or you) changes — and what stays in git

| Layer | On the VPS | Edit application code? |
|-------|------------|-------------------------|
| **App + RAG** | `git clone` from GitHub | **No** — run the released branch as-is |
| **Secrets + pilot tuning** | Copy `.env.example` → `.env` | **No** — only env vars (password hash, model, domain) |
| **HTTPS URL** | Caddy overlay + `CADDYFILE` / `SITE_ADDRESS` | **No** — config files already in the repo |
| **Firewall** | `ufw` (ports 22, 80, 443) | **No** |

A buyer’s IT team follows the same steps: provision Ubuntu + Docker, clone, configure `.env`, start Compose. No Python or Streamlit edits on the server.

---

### Phase 3 walkthrough — HTTPS pilot (recommended)

Assumes Phase 1–2 are done (SSH works, Docker installed, `ufw` allows 22/80/443).

**1. Clone and check out the deployment branch**

Until M7-6 merges to `main`, use the branch that includes Caddy:

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
git checkout feat/m7-caddy-https
```

**2. Create `.env` (only file you edit on the server)**

```bash
cp .env.example .env
nano .env
```

Minimum for a **CPX32 / 8 GB CPU** host with **client UI** and **IP + HTTPS** (no domain yet):

```bash
OLLAMA_MODEL=phi3:mini
APP_PRESENTATION_MODE=client
APP_ALLOW_DEV_TOGGLE=false
CADDYFILE=./Caddyfile.ip
```

Save and exit (`Ctrl+O`, Enter, `Ctrl+X` in nano).

Generate the TLS cert and basic-auth credentials (one time each):

```bash
chmod +x deploy/generate-ip-tls.sh deploy/generate-caddy-auth.sh
./deploy/generate-ip-tls.sh YOUR_VPS_IP
./deploy/generate-caddy-auth.sh demo 'YOUR_STRONG_PASSWORD'
```

This writes `deploy/certs/cert.pem`, `deploy/certs/key.pem`, and `deploy/caddy-basicauth.conf`. None are committed — all are gitignored.

**3. Start Ollama and pull the model**

```bash
docker compose up -d ollama
docker compose ps ollama   # wait until STATUS = healthy
docker compose exec ollama ollama pull phi3:mini
```

**4. Build and start app + Caddy**

First build downloads embedding weights and may take **15–30 minutes** on CPU.

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.caddy.yml ps
```

Expect: `ollama` **healthy**, `app` **Up**, `caddy` **Up**. Port **8501** is **not** published publicly.

**5. Open the pilot URL**

```text
https://YOUR_VPS_IP
```

Accept the browser certificate warning (self-signed IP mode). Sign in with the username and password you used in `generate-caddy-auth.sh`. Upload a PDF, wait for the green **ready** message, ask a question.

**6. Verify (M7-6 checklist)**

From your laptop:

```bash
curl -sk -o /dev/null -w "%{http_code}\n" https://YOUR_VPS_IP/_stcore/health                    # expect 401
curl -sk -o /dev/null -w "%{http_code}\n" -u demo:YOUR_PASSWORD https://YOUR_VPS_IP/_stcore/health   # expect 200
```

**Later:** add a domain → set `SITE_ADDRESS` + `ACME_EMAIL` in `.env`, switch `CADDYFILE` back to `./Caddyfile` (or remove it — that's the default), restart the Caddy overlay.

---

### 1. Prepare the server

```bash
# Example: Ubuntu/Debian — install Docker (official docs may vary by distro)
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
# Install Docker Engine 24+ and Compose plugin per https://docs.docker.com/engine/install/

sudo usermod -aG docker "$USER"
# Log out and back in so group membership applies
```

**Firewall:** allow SSH and **443** (HTTPS) from the internet when using Caddy. Do **not** expose Streamlit **8501** or Ollama **11434** publicly — the Caddy overlay keeps the app on the internal Compose network only.

```bash
# Example (ufw) — recommended with docker-compose.caddy.yml
sudo ufw allow OpenSSH
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp    # HTTP → HTTPS redirect for Let's Encrypt
sudo ufw enable
```

**Without Caddy (local dev only):** restrict port **8501** to trusted sources if you must bind it on the host:

```bash
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

For a **password-protected HTTPS URL** (recommended for pilot demos), continue with [HTTPS and basic auth (Caddy)](#https-and-basic-auth-caddy).

---

## HTTPS and basic auth (Caddy)

Use the Caddy overlay so browsers reach Streamlit over **HTTPS** with optional **HTTP basic auth**. Streamlit stays on the internal Docker network — only Caddy publishes **80** and **443** on the host.

### Prerequisites

| Item | Notes |
|------|--------|
| **DNS (production)** | `A` or `AAAA` record pointing your subdomain (e.g. `demo.example.com`) to the VPS public IP |
| **Firewall** | Allow **443** (and **80** for ACME); block public access to **8501** and **11434** |
| **`.env`** | Copy [`.env.example`](.env.example) → `.env` and set `SITE_ADDRESS` / `ACME_EMAIL` |

### 1. Configure `.env`

```bash
cp .env.example .env
```

**Production subdomain (Let's Encrypt):**

```bash
SITE_ADDRESS=demo.example.com
ACME_EMAIL=you@example.com
```

**Interim IP demo (no domain yet):** self-signed TLS on port 443 — browsers show a certificate warning; acceptable for internal sales calls until DNS is ready.

```bash
CADDYFILE=./Caddyfile.ip
```

(`SITE_ADDRESS` is not needed for `Caddyfile.ip` — Caddy listens on `:443` for all IPs.)

### 2. Generate a basic-auth credentials file

```bash
chmod +x deploy/generate-caddy-auth.sh
./deploy/generate-caddy-auth.sh demo 'CHOOSE_A_STRONG_PASSWORD'
```

This writes `deploy/caddy-basicauth.conf` (gitignored). Never commit it. Re-run to change the password.

### 3. Start the stack with Caddy

Complete [steps 3–4](#3-start-ollama-and-wait-until-healthy) (Ollama healthy + model pulled), then:

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.caddy.yml ps
```

Expect `ollama` **healthy**, `app` **Up**, and `caddy` **Up**. The app service no longer publishes **8501** on the host.

### 4. Open the app

| Setup | URL |
|-------|-----|
| **Subdomain + Let's Encrypt** | `https://demo.example.com` |
| **IP + self-signed (`Caddyfile.ip`)** | `https://YOUR_VPS_IP` (accept browser warning) |

Sign in with the username and password you set in `deploy/generate-caddy-auth.sh`.

### DNS and TLS notes

- Point **`SITE_ADDRESS`** at your VPS before first boot so Let's Encrypt can issue a certificate.
- Caddy stores certificates in the `caddy_data` volume; renewals are automatic.
- Port **8501** must **not** be reachable from the internet in this setup — verify with an external port scan or `curl` from outside the trusted network.
- When you obtain a domain, switch from `Caddyfile.ip` to the default `Caddyfile` and set `SITE_ADDRESS` + `ACME_EMAIL`.

### Verify HTTPS

On the VPS:

```bash
# Caddy admin (localhost only unless you expose it)
curl -s -o /dev/null -w "%{http_code}\n" -k https://127.0.0.1/_stcore/health
```

From your laptop (replace host):

```bash
curl -s -o /dev/null -w "%{http_code}\n" -u demo:YOUR_PASSWORD https://demo.example.com/_stcore/health
```

Expect `200`. Without credentials, the same URL should return **401**.

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

The app receives `OLLAMA_HOST=http://ollama:11434`, `USE_DUMMY_GENERATOR=false`, and `OLLAMA_MODEL=llama3.1:8b` from Compose (unless overridden in `.env`).

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
| `OLLAMA_MODEL` | `llama3.1:8b` | Override; set `phi3:mini` in `.env` for CPU-only hosts |
| `APP_ALLOW_DEV_TOGGLE` | `false` | `false` hides the developer-mode toggle (default). Set `true` only for local retrieval tuning. |
| `APP_PRESENTATION_MODE` | `client` | `client` = About + How to use only; `developer` = retrieval debug when toggle allowed |

To override via file: `cp .env.example .env`, edit values, and add `env_file: .env` under the `app` service. Generation tuning (`temperature`, `num_ctx`, etc.) follows `configs/config.yaml` → `rag.generation` unless set as `OLLAMA_*` in `.env`.

**Client demo URL:** keep `APP_ALLOW_DEV_TOGGLE=false` so buyers never see Advanced options or retrieval debug. For your own tuning on the same VPS, temporarily set `APP_ALLOW_DEV_TOGGLE=true` in `.env` and restart the app service.

See also [docs/architecture-pilot.md](docs/architecture-pilot.md#environment-variables-pilot).

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|--------|----------------|------------|
| `app` never starts / stays "Waiting" | Ollama not healthy yet | `docker compose logs ollama`; wait for `healthy` in `docker compose ps`. Slow disks may need more time within the healthcheck `start_period`. |
| **Connection refused** to Ollama | Ollama down, wrong host, or app started before Ollama was ready | Use Compose (not `docker run` app alone). On VPS, app must use `OLLAMA_HOST=http://ollama:11434` — not `localhost`. Check `docker compose ps` shows `ollama` **healthy**. |
| **Connection refused** on `YOUR_VPS_IP:8501` | Firewall, wrong IP, app not running, or Caddy overlay active | With Caddy, use `https://YOUR_VPS_IP` (not `:8501`). Without Caddy, `docker compose ps` and open 8501 only to trusted CIDRs; confirm `curl http://127.0.0.1:8501/_stcore/health` on the VPS returns 200. |
| **401** on HTTPS URL with correct password | Wrong hash in conf file | Regenerate: `./deploy/generate-caddy-auth.sh demo 'your-password'`, then restart Caddy. |
| **`ERR_SSL_PROTOCOL_ERROR` / TLS internal error (592)** | Missing cert files, wrong IP in SAN, or stale Caddy cache | Re-run `./deploy/generate-ip-tls.sh YOUR_VPS_IP`, then stop Caddy, remove its volumes (`docker volume rm …_caddy_data …_caddy_config`), and recreate. Note: `tls internal` does not work for bare IP addresses — explicit cert files are required. |
| **Certificate / TLS errors** | DNS not propagated, or IP demo self-signed cert | For subdomains, wait for DNS and ensure ports 80/443 reach Caddy. For `Caddyfile.ip`, accept the browser warning or switch to a real subdomain. |
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
- Pilot demos should use [HTTPS and basic auth (Caddy)](#https-and-basic-auth-caddy) so Streamlit is not exposed on plain HTTP to the internet.
