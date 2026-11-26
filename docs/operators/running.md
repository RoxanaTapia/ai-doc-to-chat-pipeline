# Running the stack

> **For clients:** the [live pilot](https://ai-doc-pilot.roxanatapia.dev/) is the product. This page is for operators and contributors.

## Background

This repo runs the PDF Q&A app. Put the app behind an existing reverse proxy when you want public HTTPS.

> **Takeaway:** Compose here is `app` (optional `api`). The reverse proxy in front owns the public ports. The answer model is a hosted API.

Python 3.12 or newer. The answer writer is Claude by default. OpenAI and Grok are the other choices.

---

## Local

Clone the repository, then copy the example env file:

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
cp .env.example .env
```

Set one answer model in `.env`, then pass `--env-file .env`. A value only in the shell is not enough.

| `LLM_PROVIDER` | Model | Key |
|----------------|-------|-----|
| `anthropic` (empty, dummy mode off) | Claude Haiku 4.5 | `ANTHROPIC_API_KEY` |
| `openai` | GPT-4.1 mini | `OPENAI_API_KEY` |
| `xai` | Grok 4.7 | `XAI_API_KEY` |

`LLM_PROVIDER=xai` with no `XAI_API_KEY` stops the answer and names that missing key. The same pattern applies to `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. Leave `LLM_PROVIDER` empty and set `USE_DUMMY_GENERATOR=true` for placeholder answers with no key.

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up --build -d
```

The demo listens on port **8501**, path **`/app`**: [http://localhost:8501/app](http://localhost:8501/app).

On start, the app loads the sample NDA embeddings and search index before it opens that port. The first question uses the ready index.

Stop the stack:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml down
```

Optional thin API (OpenAPI at `/docs`):

```bash
docker compose --env-file .env -f deploy/docker-compose.yml --profile api up --build -d api
```

### Local Python

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run apps/demo/app.py
```

On Windows, activate with `.venv\Scripts\activate`. The same port and path apply. Put the provider and its key in `.env`. The comments in `.env.example` list the usual values.

---

## Behind a reverse proxy

This app does not terminate TLS. An existing reverse proxy in front does. The optional overlay joins an external Docker network named `edge` and does not publish the app port.

```bash
docker network create edge   # once; ignore the error if it already exists

docker compose --env-file .env -p ai-doc-to-chat-pipeline \
  -f deploy/docker-compose.yml -f deploy/docker-compose.shared-edge.yml up -d
```

Expect: `app` up. The app port stays unpublished.

| Alias | Port | Role |
|-------|------|------|
| `app` | 8501 | PDF Q&A (Streamlit) |

Point the reverse proxy at `app:8501`. Streamlit's `baseUrlPath` is `app` (`.streamlit/config.toml`), so keep that prefix when the proxy serves the app under `/app`. The proxy owns `/`. The UI entrypoint is `apps/demo/app.py`.

Keep `COMPOSE_PROJECT_NAME=ai-doc-to-chat-pipeline` so Compose project names stay stable. Do not copy this example over a filled `.env`.

---

## Demo limits

The public demo answers from the sample NDA. Visitors cannot upload a PDF unless you turn that on. A signed-in operator can upload either way. The page states the largest size it will accept, and refuses a bigger file.

Model calls pause when one visitor has asked enough for the hour, or when the whole demo has asked enough for the day. A question the document cannot answer does not count, because the model is not called. The page then says: "This demo has reached its limit for now. Please try again later, or book a short call and I'll walk you through it."

The counts are kept in a small file so a refresh does not reset them. Compose stores that file in the `demo-usage` volume, which stays when you recreate the app container.

| Setting | Empty means | What it does |
|---------|-------------|--------------|
| `DEMO_ALLOW_UPLOAD` | off | `true` lets a visitor upload a PDF |
| `DEMO_UPLOAD_MAX_MB` | 10 | Largest PDF, in megabytes. The page states this size |
| `DEMO_VISITOR_HOURLY_CAP` | 20 | Model calls one visitor may make in an hour |
| `DEMO_DAILY_CAP` | 200 | Model calls the whole demo may make in one day, from midnight UTC |
| `DEMO_USAGE_STORE` | `data/usage.sqlite` | File that remembers those counts |

The visitor is the address in `X-Forwarded-For`, the one the reverse proxy added. If that header is missing, the hour follows this browser session. Leave a value empty in `.env` to keep the default. Do not commit a filled `.env`.

---

## Pin a version

The core package is `ai-doc` **1.0.0** ([`pyproject.toml`](../../pyproject.toml), and `ai_doc.__version__`). The demo is not its own package. Streamlit is the `demo` extra.

Release tags are named `vX.Y.Z`. Pin this release with:

```bash
pip install "git+https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline@v1.0.0"
```

Add the demo extra when you need the Streamlit demo:

```bash
pip install "ai-doc[demo] @ git+https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline@v1.0.0"
```

From a checkout of that tag, `pip install .` installs the core, and `python -c "import ai_doc; print(ai_doc.__version__)"` prints `1.0.0`.
