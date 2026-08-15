# Private PDF Q&A

> Grounded answers from a confidential PDF, with page citations, on infrastructure you control.

Upload a policy, SOP, report, or contract. Ask in plain language. Every answer cites the page it used. Documents stay on **your server**.

> **Takeaway:** This is a retrieval system, not a chatbot that guesses. If the document cannot answer, it says so.

---

## Try the pilot

🟢 **Live** · <a href="https://ai-doc-pilot.roxanatapia.dev/" target="_blank" rel="noopener noreferrer"><strong>ai-doc-pilot.roxanatapia.dev</strong></a>

Real answers, HTTPS, invite-protected. [Request a walkthrough on Upwork](https://www.upwork.com/freelancers/roxanadev). Once you have access, try the [sample NDA](docs/product/sample-nda.pdf).

Uploaded files are processed in memory and never stored. Each session starts fresh. Use only sample or non-confidential documents on the shared pilot.

---

## How an answer is grounded

```mermaid
flowchart LR
  A[Upload PDF] --> B[Extract]
  B --> C[Chunk]
  C --> D[Embed locally]
  D --> E[Hybrid search]
  E --> F[Rerank]
  F --> G[Cited answer]
```

| Step | What happens |
|------|----------------|
| **Ingest** | PyMuPDF reads the PDF. Scanned pages go through OCR. Page numbers stay on every passage. |
| **Chunk** | Text is split at section headers so clauses do not bleed across chunks. |
| **Embed** | Local sentence-transformers. Vectors never leave the server. |
| **Retrieve** | Hybrid search: dense (FAISS) plus BM25, fused with reciprocal rank fusion. |
| **Rerank** | A cross-encoder scores the shortlist so the model sees the best passages first. |
| **Cite or refuse** | The answer shows page and excerpt. If the document does not contain the answer, the system says so. |

How that was tested: [pilot evaluation](docs/product/pilot-evaluation.md).

---

## What it does well

- **Policies, SOPs, reports, contracts:** find definitions, obligations, dates, and rules in the PDF you uploaded
- **Sourced answers:** every response cites the page and excerpt it used
- **Private by design:** local embeddings; local LLM for air-gap, or a swappable API model for demos
- **Honest when empty:** refuses to invent a clause that is not in the document

## Known limits

- **Session-based:** re-upload after restart; no shared document library yet
- **One PDF at a time:** not enterprise search across a file store
- **Read, don't calculate:** finds printed numbers; does not sum or verify math
- **Document Q&A:** answers questions about the uploaded file; does not connect to CRM, email, or ticketing

---

## Stack

PyMuPDF · FAISS · BM25 · bge-reranker · sentence-transformers · Streamlit · FastAPI · Ollama or Anthropic

MIT licensed · [Roxana Tapia](https://github.com/RoxanaTapia) · [Upwork](https://www.upwork.com/freelancers/roxanadev)
