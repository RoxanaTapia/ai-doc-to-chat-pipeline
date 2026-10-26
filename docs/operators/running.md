# Running the stack

> **For clients:** the [live pilot](https://ai-doc-pilot.roxanatapia.dev/) is the product. This page is for operators and contributors.

## Background

This repo runs the PDF Q&A app and the model. Put the app behind an existing reverse proxy when you want public HTTPS.

> **Takeaway:** Compose here is `app` + `ollama` (optional `api`). The reverse proxy in front owns the public ports.

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

Host venv instead of Docker for the UI: `pip install -r requirements.txt` then `PYTHONPATH=src streamlit run apps/demo/app.py`. Point `OLLAMA_HOST` at a reachable Ollama.

---

## Behind a reverse proxy

This app does not terminate TLS. An existing reverse proxy in front does. The optional overlay joins an external Docker network named `edge` and does not publish the app port.

```bash
docker network create edge   # once; ignore the error if it already exists

docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.shared-edge.yml up -d
```

Expect: `ollama` healthy, `app` up. The app port stays unpublished.

| Alias | Port | Role |
|-------|------|------|
| `app` | 8501 | PDF Q&A (Streamlit) |

Point the reverse proxy at `app:8501`. Streamlit's `baseUrlPath` is `app` (`.streamlit/config.toml`), so keep that prefix when the proxy serves the app under `/app`. The proxy owns `/`. The UI entrypoint is `apps/demo/app.py`.

Keep `COMPOSE_PROJECT_NAME=ai-doc-to-chat-pipeline` so Ollama volume names stay stable. Do not copy this example over a filled `.env`.
