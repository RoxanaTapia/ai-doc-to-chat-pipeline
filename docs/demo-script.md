# Demo video — 5-minute storyboard

Public outline for a sales walkthrough of the live pilot. A full recording script with pre-flight checks, operator notes, and extended Q&A lives in local **`docs-private/demo-script.md`** (gitignored — not in this repo).

## Audience

Technical buyer or evaluator who saw the [public Streamlit demo](https://ai-doc-to-chat-demo.streamlit.app) and wants proof of real, grounded answers on infrastructure you control.

## Before you record

- Live pilot reachable at your password-protected URL — see [DEPLOYMENT.md](../DEPLOYMENT.md)
- [Sample NDA](sample-nda.pdf) ready to upload
- Ollama warm (one throwaway question after login)
- Browser window ~1280×720; hide bookmarks bar and unrelated tabs

## Storyboard (~5 min)

| Time | Scene | Show | Say (gist) |
|------|-------|------|------------|
| 0:00–0:30 | Hook | Public demo dummy-mode banner → open live pilot login | “The free demo is UI-only; here is the same app with a real local LLM on a private VM.” |
| 0:30–1:00 | Trust | HTTPS lock, basic-auth login, in-app privacy note | “Documents stay in memory for the session — nothing is stored on the shared pilot.” |
| 1:00–2:00 | Upload | Upload sample NDA; expand extracted-text preview | “Upload a PDF; the app extracts text, chunks it, and builds a searchable index in-process.” |
| 2:00–3:30 | Q&A | Two or three grounded questions with sources visible | Ask about parties, term, or a specific clause; point to page citations in the answer. |
| 3:30–4:30 | Limits | One question the model should refuse or hedge on | Show honest “not in document” or session scope — contrast with generic chatbots. |
| 4:30–5:00 | Close | [Architecture diagram](architecture-pilot.md) or deployment guide | “Same Compose stack deploys on your VPS; next step is evaluation on your documents.” |

## Sample questions (sample NDA)

Use questions that match numbered sections so citations are obvious:

1. Who are the parties to this agreement?
2. What is the term or duration?
3. What obligations apply to confidentiality?

Expected answer quality and tuning notes: [pilot evaluation](pilot-evaluation.md).

## After recording

When the walkthrough is published (Loom, YouTube unlisted, etc.), add the URL to the **Demo video** line in [README.md](../README.md). No other code change required until the link exists.
