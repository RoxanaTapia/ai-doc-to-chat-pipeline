# Architecture — single-VM pilot (M7 target)

Reference layout for the self-hosted demo / small-office pilot.

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
         │  FAISS + embeddings (in-process, M7)
         ▼
   User browser (chat + sources)
```

## Data flow (one question)

1. User uploads PDF → PyMuPDF (+ optional OCR) → chunks → FAISS index (session memory).
2. User asks question → semantic/hybrid retrieval → context assembly.
3. App POSTs prompt to Ollama → grounded answer + source chunks in UI.

## M9+ evolution

Replace in-memory FAISS with **Postgres/pgvector** and **MinIO** for PDF storage;
add **FastAPI** (M8) as the integration surface. Ollama or Anthropic (M12) as LLM backend.

## Environment variables (pilot)

| Variable | Purpose |
|----------|---------|
| `OLLAMA_HOST` | Ollama URL (e.g. `http://ollama:11434` in Compose) |
| `USE_DUMMY_GENERATOR` | `false` for real generation |
| `OLLAMA_MODEL` | Optional override (default from `configs/config.yaml`) |
