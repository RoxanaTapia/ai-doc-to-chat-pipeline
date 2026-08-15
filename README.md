# Private PDF Q&A

> Grounded answers from a confidential PDF, with page citations, on infrastructure you control.

Upload a policy, SOP, report, or contract. Ask in plain language. Every answer cites the page it used. Documents stay on **your server**.

> **Takeaway:** This is a retrieval system, not a chatbot that guesses. If the document cannot answer, it says so.

---

## Try the pilot

🟢 **Live** · <a href="https://ai-doc-pilot.roxanatapia.dev/" target="_blank" rel="noopener noreferrer"><strong>ai-doc-pilot.roxanatapia.dev</strong></a>

- Request an invite on the gate, then upload a PDF and ask a question
- Start with the [sample NDA](docs/product/sample-nda.pdf) if you want a ready document
- Optional: [ask for a walkthrough on Upwork](https://www.upwork.com/freelancers/roxanadev)

Uploaded files are processed in memory and never stored. Each session starts fresh. Use only sample or non-confidential documents on the shared pilot.

---

## How an answer is grounded

Each step exists to keep the answer inside the PDF you uploaded.

```mermaid
flowchart LR
  A[Upload PDF] --> B[Extract]
  B --> C[Chunk]
  C --> D[Embed locally]
  D --> E[Hybrid search]
  E --> F[Rerank]
  F --> G[Cited answer]
```

| Step | Technique | Why it matters |
|------|-----------|----------------|
| **Ingest** | PyMuPDF, plus OCR on scanned pages | Text and page number travel together, so a citation can point at a real page. |
| **Chunk** | Split at section headers | Nearby clauses stay separate. A question about Section 3 should not pull the end of Section 2. |
| **Embed** | Local sentence-transformers | Search by meaning without sending the document to an outside embedding API. |
| **Retrieve** | Hybrid search: FAISS + BM25, fused with RRF | Embeddings catch paraphrases. Keyword search catches exact terms such as section numbers. |
| **Rerank** | Cross-encoder (`bge-reranker`) | A second pass reads the question and each passage together, so the model sees the best evidence first. |
| **Cite or refuse** | Page + excerpt, or "not in the document" | You can check the source. If retrieval found too little, the system does not invent a clause. |

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
