# Running the stack

> **For clients:** the [live pilot](https://ai-doc-pilot.roxanatapia.dev/) is the product. This page is for operators and contributors.

## Background

This repo runs the PDF Q&A app and Ollama. HTTPS, the invite gate, and TLS live in a private shared-edge config (walkthrough on request).

> **Takeaway:** Compose here is `app` + `ollama` (optional `api`). Public ports 80/443 are not this tree.

---

## Local

From the repo root:

```bash
cp .env.example .env
docker compose -f deploy/docker-compose.yml up --build -d
docker compose -f deploy/docker-compose.yml exec ollama ollama pull phi3:mini
```

Open [http://localhost:8501](http://localhost:8501). The app binds 8501 on the host in this file.

Optional thin API (OpenAPI at `/docs`):

```bash
docker compose -f deploy/docker-compose.yml --profile api up --build -d api
```

Host venv instead of Docker for the UI: `pip install -r requirements.txt` then `streamlit run src/app.py`. Point `OLLAMA_HOST` at a reachable Ollama.

---

## Live pilot (shared host)

A private shared-edge config (walkthrough on request) terminates TLS and runs the invite gate. This app does not. Join Docker network `edge` and leave ports 80 and 443 alone.

```bash
docker network create edge   # once; ignore the error if it already exists

docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.shared-edge.yml up -d
```

Expect: `ollama` healthy, `app` up, **no** `caddy` service. The host does not publish 8501.

| Alias | Port | Role |
|-------|------|------|
| `app` | 8501 | PDF Q&A (Streamlit) |

Caddy forwards `/app*` to `app:8501` and **keeps** the `/app` prefix. Streamlit's `baseUrlPath` is `app` (`.streamlit/config.toml`). The public gate owns `/`.

Keep `COMPOSE_PROJECT_NAME=ai-doc-to-chat-pipeline` so Ollama volume names stay stable. The shared-edge stack uses its own compose project name and reads the `.env` next to the compose file for invite and TLS keys. Those keys are not in this repo's `.env.example`. **Do not overwrite the server `.env` with the example file.**

Invite minting and TLS reloads happen in that shared-edge config. Confirm that stack is up before recreating `app`.
