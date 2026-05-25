# Pilot evaluation

Ongoing evaluation of the RAG stack on NDA-style contracts. Each round tests a specific stack change against a fixed question set and gold standard. Full round-by-round detail is in [`docs/eval-history/`](eval-history/).

---

## Question set

Consistent across all rounds on the same 4-page sample NDA:

| # | Question | Type |
|---|----------|------|
| **Q1** | How is Confidential Information defined in Section 1? | Definition lookup |
| **Q2** | What obligations does Section 3 impose on the Receiving Party? | Obligation list |
| **Q3** | Does this agreement specify liquidated damages or a fixed financial penalty? | Honesty trap (answer is No) |

Sample document: [`docs/sample-nda.pdf`](sample-nda.pdf)

---

## Progression across rounds

| Round | Stack highlight | Q1 | Q2 | Q3 | Status |
|-------|----------------|----|----|-----|--------|
| **1** | `phi3:mini` · baseline retrieval | Partial | Partial | Pass | Section bleed; Q3 honesty confirmed |
| **2** | `llama3.1:8b` · hybrid retrieval · prompt v2 | Partial | Partial | Pass | Better structure; source ranking still noisy |
| **3** | Reranker · section metadata · prompt v3 | Partial+ | Pass | Pass | Q2 section alignment improved |
| **4** | Hard context filter (R4-E) · section-aware boost | Partial+ | Pass | Pass | Q2 on-section 1/1; Q3 honesty held |
| **5** | Header-aware chunking · dedupe · context guard | Pass | Pass | Pass | Generalises to 2-page unnumbered NDA |

**Current bar (Round 5):** Q1–Q3 Pass on sample NDA · Q2/Q3 Pass on client-style 2-page NDA · demo-ready.

---

## Round 5 — full detail (latest)

**Status:** Complete — May 2026  
**Stack:** Docker Compose · `llama3.1:8b` · hybrid · `top_k: 5` · reranker ON · section metadata ON · hard context filter ON · `chunk_overlap: 200` · `split_on_legal_headers: true` · dedupe ON · context sufficiency guard ON

### Verdict

| Document | Q1 | Q2 | Q3 | Overall |
|----------|----|----|-----|---------|
| **Sample NDA** (4 pp, numbered sections) | Pass / Partial+ · 1/1 on-section | Pass · four duties · 1/1 on-section | Pass · honest No | **Go — demo-ready** |
| **Client-style 2-page NDA** (title headers) | _(not re-run post-tuning)_ | Pass · four duties incl. copy/pass-on | Pass · honest No | **Go for workflow + honesty demo** |

### Tuning shipped (R5)

| ID | Change | Effect |
|----|--------|--------|
| #1 | Prompt v4 — no invented section numbers; one bullet per obligation line | Honest citations on unnumbered docs |
| #2 | Header-aware chunking — split at `N.` and `Title.` legal headers | Confidentiality block isolated on 2-page NDA |
| #3 | Near-duplicate drop + context sufficiency guard + dev sidebar defaults | Removed false-positive rank-1 "Agreement." chunk |

### Sample NDA results (post-tuning)

| Q | Content | Context purity | Latency |
|---|---------|----------------|---------|
| **Q1** | Pass / Partial+ — definition + examples | 1/1 on-section · ~498 chars | 31.9s |
| **Q2** | Pass — four duties incl. return/destroy | 1/1 on-section · ~600 chars | 27.9s |
| **Q3** | Pass — honest No; injunctive relief only; no invented €/$ | N/A (no penalty clause) · ~2514 chars | 44.1s |

### Client-style 2-page NDA results

| Q | Content | Sources | Latency |
|---|---------|---------|---------|
| **Q2** | Pass — four duties incl. discuss/copy/pass-on | Source 1 = Confidentiality block (rel 1.0) | ~25–32s warm |
| **Q3** | Pass — clear No; no fixed amount invented | Irrelevant sources expected (no penalty clause) | ~32s warm |

**Baseline (pre-tuning #2):** Q2 failed — obligation bullets not in context; generic "keep confidential" only. Header split fixed this.

### VPS smoke (phi3:mini, num_ctx=1024)

Verified live on `ai-doc-pilot.roxanatapia.dev`:

| Q | Result | Latency |
|---|--------|---------|
| Q1 | Pass | 37.2s |
| Q2 | Pass | 15.8s |
| Q3 | Partial — honest No but verbose hedging | 41s |

Q3 hedging is a `phi3:mini` formatting artifact (not a hallucination). `llama3.1:8b` gives a cleaner answer.

---

## Deferred

- Full Q1 re-run on 2-page NDA post-tuning (Partial+ pre-tuning; lower priority)
- 10–40 page client PDF evaluation
- Demo video (M7-7)

---

## Demo guidance

| Use case | Document |
|----------|---------|
| Section-number questions (Q1/Q2) + context purity story | Sample NDA |
| Obligation completeness + Q3 honesty on unnumbered layout | 2-page NDA |
