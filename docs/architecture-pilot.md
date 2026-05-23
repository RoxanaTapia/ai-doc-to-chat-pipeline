# Architecture — single-VM pilot

Reference layout for a self-hosted demo or small-office pilot.

```text
                    Internet
                        │
                        ▼
              ┌─────────────────┐
              │  Caddy / nginx  │  HTTPS + basic auth (optional)
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  ┌──────────────┐           ┌──────────────┐
  │  Streamlit   │  HTTP     │    Ollama    │
  │  (app :8501) │ ────────► │  (:11434)    │
  └──────────────┘ OLLAMA_HOST└──────────────┘
         │
         │  PDF upload → extract → chunk
         │  FAISS + embeddings (in-process)
         ▼
   User browser (chat + sources)
```

## Data flow (one question)

1. User uploads PDF → PyMuPDF (+ optional OCR) → chunks → FAISS index (session memory).
2. User asks question → semantic/hybrid retrieval → context assembly.
3. App POSTs prompt to Ollama → grounded answer + source chunks in UI.

## Planned evolution

Production rollouts typically add persistent document storage, a REST API for integrations, and enterprise authentication — while keeping the same retrieval and generation core. Ollama remains the default LLM; cloud APIs can be swapped in where procurement requires it.

## Environment variables (pilot)

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama URL (e.g. `http://ollama:11434` in Compose) |
| `USE_DUMMY_GENERATOR` | `false` for real generation |
| `OLLAMA_MODEL` | Optional override (default from `configs/config.yaml`) |

Full operator notes: see [DEPLOYMENT.md](../DEPLOYMENT.md#environment-variables).
