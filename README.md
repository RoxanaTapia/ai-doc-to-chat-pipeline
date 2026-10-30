# PDF Q&A

> Grounded answers from a confidential PDF.

My pilot shows the best-matching page next to each answer and says when the document doesn't cover the question.

Upload a policy, SOP, report, or contract. Ask in plain language.

> **Takeaway:** This is a retrieval system, not a chatbot that guesses.

[Demo video](https://youtu.be/DdIm7N0kFx0)

---

## Try the pilot

🟢 **Live** · <a href="https://ai-doc-pilot.roxanatapia.dev/" target="_blank" rel="noopener noreferrer"><strong>ai-doc-pilot.roxanatapia.dev</strong></a>

- Request an invite on the gate, then upload a PDF and ask a question
- Start with the [sample NDA](docs/product/sample-nda.pdf) if you want a ready document
- Optional: [ask for a walkthrough on Upwork](https://www.upwork.com/freelancers/roxanadev)

Only the few passages needed for an answer are sent to a hosted model, Anthropic (Claude). Your upload is cleared when you leave or reset. Each session starts fresh. Use only sample or non-confidential documents on the shared pilot.

---

## What it does well

| Feature | In practice |
|---------|-------------|
| **Best-matching page** | You see the page that matched, next to the answer |
| **Answer writer** | Anthropic (Claude) or Ollama |
| **Embeddings** | Search runs with the app |
| **Honest refusals** | You are told when the PDF does not contain the answer |
| **Document types** | Policies, SOPs, reports, contracts |

## Known limits

| Limit | In practice |
|-------|-------------|
| **Session only** | You re-upload after a restart. There is no shared library yet |
| **One PDF** | You search one file, not a file store |
| **No math** | You get the numbers printed in the PDF. The app does not calculate |
| **Q&A only** | No CRM, email, or ticketing |

---

## What is in this repo

The retrieval core and the Streamlit demo are separate. You can install the core on its own.

| Piece | Path | What it is |
|-------|------|------------|
| **Core** | [`src/ai_doc`](src/ai_doc/) | Python package `ai-doc`. Ingest, chunk, search, and the answer. Optional HTTP API in [`src/ai_doc/api`](src/ai_doc/api/). |
| **Demo** | [`apps/demo`](apps/demo/) | Streamlit client: you upload a PDF, ask, and see the page. The core does not import it. |

| Path | What it is |
|------|------------|
| [`configs/`](configs/) | Chunk size, retrieval, and prompts |
| [`deploy/`](deploy/) | App image and Compose: the demo, Ollama, and an optional API |
| [`docs/`](docs/README.md) | Product and operator pages |
| [`.env.example`](.env.example) | Product knobs. Copy to `.env`. Never commit a filled file. |

How an answer is produced: [docs/README.md](docs/README.md). Run the stack: [docs/operators/running.md](docs/operators/running.md). Tests and review: [AGENTS.md](AGENTS.md).

---

## Support

Questions are welcome.

| Need | Where |
|------|--------|
| **Bugs and product questions** | [Open an issue](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues) |
| **A walkthrough of the pilot** | [Upwork](https://www.upwork.com/freelancers/roxanadev), also linked under [Try the pilot](#try-the-pilot) |
| **How changes are reviewed** | [How I work with AI](docs/how-we-work.md) |

The core uses semantic versions. Fixes go to the latest minor version of `ai-doc`. Pin an older commit when that exact tree should stay put. Install commands: [running the stack](docs/operators/running.md).

---

MIT licensed · [Roxana Tapia](https://github.com/RoxanaTapia) · [Upwork](https://www.upwork.com/freelancers/roxanadev) · [How I work with AI](docs/how-we-work.md)

What changed, which versions get fixes, and how to report a problem: [changelog](CHANGELOG.md), [support](SUPPORT.md), [security](SECURITY.md).
