# Architecture — single-VM pilot

Reference layout for a self-hosted evaluation or small-office pilot. Matches the Compose files in this repo: `docker-compose.yml` plus optional `docker-compose.caddy.yml` for HTTPS.

## Stack overview

```text
                    Internet (HTTPS :443, HTTP :80)
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Caddy 2.9            │
                    │  TLS (Let's Encrypt)  │
                    │  Basic auth           │
                    └───────────┬───────────┘
                                │ reverse_proxy → app:8501
                                ▼
                    ┌───────────────────────┐
                    │  Streamlit app        │
                    │  :8501 (internal)     │
                    │  PDF → chunk → FAISS  │
                    │  (in-process / RAM)   │
                    └───────────┬───────────┘
                                │ OLLAMA_HOST
                                ▼
                    ┌───────────────────────┐
                    │  Ollama               │
                    │  :11434 (internal)    │
                    │  volume: ollama_models│
                    └───────────────────────┘

  User browser ◄── chat UI + source citations (HTTPS via Caddy only)
```

With the Caddy overlay active, Streamlit and Ollama are **not** published to the host — only ports `80` and `443` are exposed.

## Compose layout

| Service | Image / build | Published ports | Role |
|---------|---------------|-----------------|------|
| `caddy` | `caddy:2.9.1-alpine` | `80`, `443` | TLS termination, basic auth, reverse proxy |
| `app` | Dockerfile (this repo) | *(none with Caddy overlay)* | Streamlit UI, RAG pipeline, local embeddings |
| `ollama` | `ollama/ollama:0.6.5` | *(internal)* | Local LLM inference; models on `ollama_models` volume |

Production pilot start command:

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d
```

For local development without HTTPS, use `docker-compose.yml` alone — `app` then binds `8501` on the host.

## Data flow (one question)

1. User uploads PDF in browser → PyMuPDF (+ optional OCR) → text chunks → in-memory FAISS index.
2. User asks a question → semantic/hybrid retrieval → context assembled with page metadata.
3. App POSTs prompt to Ollama over the Compose network → grounded answer + source excerpts in UI.

Documents and vectors live in the app process for the session; only Ollama model weights persist on disk (`ollama_models` volume).

## Planned evolution

Production rollouts typically add persistent document storage, a REST API for integrations, and enterprise authentication — while keeping the same retrieval and generation core. Ollama remains the default LLM; cloud APIs can be swapped in where procurement requires it.

## Environment variables (pilot)

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama URL (e.g. `http://ollama:11434` in Compose) |
| `USE_DUMMY_GENERATOR` | `false` for real generation |
| `OLLAMA_MODEL` | Model tag (e.g. `phi3:mini` on CPU hosts) |
| `SITE_ADDRESS` / `ACME_EMAIL` | Domain and email for Let's Encrypt (Caddy overlay) |

Full operator notes: [DEPLOYMENT.md](../DEPLOYMENT.md#environment-variables).
