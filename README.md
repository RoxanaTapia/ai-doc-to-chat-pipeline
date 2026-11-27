# PDF Q&A

> Grounded answers from a confidential PDF.

My pilot shows the best-matching page next to each answer and says when the document doesn't cover the question.

The Beta Labs sample NDA is already loaded. Ask in plain language. Only a signed-in operator can upload a PDF.

> **Takeaway:** This is a retrieval system, not a chatbot that guesses.

---

## What works now

<span id="known-limits"></span>

| What you can do | In practice |
|-----------------|-------------|
| **One PDF** | Your questions cover that file |
| **Search** | Search runs with the app |
| **Printed numbers** | You get the numbers printed in the PDF. The app does not calculate |
| **Questions and answers** | You do not get a CRM, email, or ticketing |

## How it fits together

Each step keeps the answer inside the Beta Labs sample NDA, or inside a PDF a signed-in operator uploaded. The step-by-step picture is in the [docs map](docs/README.md).

## What's planned

A shared library, sign-in for your organization, and a separate space per project wait until you want them for a real engagement. The way the pilot finds an answer stays the same.

## Try it

<span id="try-the-pilot"></span>

🟢 **Live** · <a href="https://ai-doc-pilot.roxanatapia.dev/" target="_blank" rel="noopener noreferrer"><strong>ai-doc-pilot.roxanatapia.dev</strong></a>

▶ **[Watch the pilot](https://youtu.be/DdIm7N0kFx0)**

- Request an invite on the gate, then ask a question about the Beta Labs sample NDA
- The file itself is the [sample NDA](docs/product/sample-nda.pdf)
- Optional: [ask for a walkthrough on Upwork](https://www.upwork.com/freelancers/roxanadev)

Only the few passages needed for an answer are sent to a hosted model. Claude is the default. OpenAI and Grok are the other choices. Visitors on the shared pilot stay on the sample NDA. When the pilot has answered enough questions for the hour or the day, it asks you to try again later or book a short call. A PDF a signed-in operator uploads can be no larger than the size shown on the page, and is cleared when they leave or reset. Each session starts fresh. Use only sample or non-confidential documents on the shared pilot.

## Where things are

| Path | What you find |
|------|----------------|
| [`src/ai_doc`](src/ai_doc/) | The core you can install on its own. It reads the PDF, searches, and produces the answer. Optional HTTP API in [`src/ai_doc/api`](src/ai_doc/api/). |
| [`apps/demo`](apps/demo/) | What you open for the pilot |
| [`configs/`](configs/) | Chunk size, search, and prompts |
| [`deploy/`](deploy/) | The files that package the pilot |
| [`docs/`](docs/README.md) | Product and operator pages, including the [board](docs/board.md) and the [goal](docs/goal.md) |
| [`.env.example`](.env.example) | Product knobs. Copy to `.env`. Never commit a filled file |

The core uses semantic versions. Fixes go to the latest minor version of `ai-doc`. Pin an older commit when that exact tree should stay put. Details are in [support](SUPPORT.md). Install commands are in [running the stack](docs/operators/running.md).

What changed: [changelog](CHANGELOG.md). Report a vulnerability: [security](SECURITY.md). Bugs and product questions: [open an issue](https://github.com/RoxanaTapia/ai-doc-to-chat-pipeline/issues). Tests and review: [AGENTS.md](AGENTS.md).

---

[MIT licensed](LICENSE) · [Roxana Tapia](https://github.com/RoxanaTapia) · [How I work with AI](docs/how-we-work.md)
