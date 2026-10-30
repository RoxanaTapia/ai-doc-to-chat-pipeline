# Running the stack

> **For clients:** the [live pilot](https://ai-doc-pilot.roxanatapia.dev/) is the product. This page is for operators and contributors.

## Background

This repo runs the PDF Q&A app and the model. Put the app behind an existing reverse proxy when you want public HTTPS.

> **Takeaway:** Compose here is `app` + `ollama` (optional `api`). The reverse proxy in front owns the public ports.

Python 3.12 or newer. The answer writer is Ollama or Anthropic (Claude).

---

## Local

Clone the repository, then copy the example env file:

```bash
git clone https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline.git
cd ai-doc-to-chat-pipeline
cp .env.example .env
```

An empty `OLLAMA_MODEL` selects `llama3.1:8b`. Append `phi3:mini` and pass `--env-file .env` so the model you pull is the one the app calls. A value only in the shell is not enough. Both tags are Ollama models.

```bash
printf '\nOLLAMA_MODEL=phi3:mini\n' >> .env
docker compose --env-file .env -f deploy/docker-compose.yml up --build -d
docker compose --env-file .env -f deploy/docker-compose.yml exec ollama ollama pull phi3:mini
```

The demo listens on port **8501**, path **`/app`**: [http://localhost:8501/app](http://localhost:8501/app).

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

On Windows, activate with `.venv\Scripts\activate`. The same port and path apply. Set `OLLAMA_HOST` in `.env` to an Ollama you can reach. The comments in `.env.example` list the usual values.

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
