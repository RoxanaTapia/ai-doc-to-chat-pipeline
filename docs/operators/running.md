# Running the stack

> **For clients:** the [live pilot](https://ai-doc-pilot.roxanatapia.dev/) is the product. This page is for operators and contributors.

## Background

This repo runs the PDF Q&A app and Ollama. HTTPS, the invite gate, and TLS live in [roxanatapia-edge](https://github.com/RoxanaTapia/roxanatapia-edge).

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

## Live pilot (portfolio VPS)

Caddy and invites run in `roxanatapia-edge`. This app joins Docker network `edge` as alias `app`:

```bash
docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.shared-edge.yml up -d
```

Keep `COMPOSE_PROJECT_NAME=ai-doc-to-chat-pipeline` so Ollama volume names stay stable. Edge Compose still reads `/root/ai-doc-to-chat-pipeline/.env` for invite and TLS keys. Those keys are not in this repo's `.env.example`. **Do not overwrite the server `.env` with the example file.**

After a pull, `deploy/invite/` and this repo's Caddy overlay are gone. Mint and Caddy reloads happen in [roxanatapia-edge](https://github.com/RoxanaTapia/roxanatapia-edge). Confirm the edge stack is up and that this project is not running a second Caddy.

Streamlit serves under `/app` (`baseUrlPath` in `.streamlit/config.toml`) because the public gate owns `/`.
