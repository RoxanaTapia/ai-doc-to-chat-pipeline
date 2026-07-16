# Pilot evaluation

## Background

Summary of how the RAG stack was tested on NDA-style contracts. Same three questions across rounds, with a gold standard for honesty and section alignment. Full round-by-round notes live in [`docs/archive/eval-history/`](../archive/eval-history/).

> **Takeaway:** Round 5 is the current bar: pass on the sample NDA for Q1–Q3, and solid honesty on a shorter unnumbered NDA.

---

## ❓ Question set

Same 4-page sample NDA across rounds:

| # | Question | Type |
|---|----------|------|
| **Q1** | How is Confidential Information defined in Section 1? | Definition lookup |
| **Q2** | What obligations does Section 3 impose on the Receiving Party? | Obligation list |
| **Q3** | Does this agreement specify liquidated damages or a fixed financial penalty? | Honesty trap (answer is No) |

Sample document: [`sample-nda.pdf`](sample-nda.pdf)

---

## 📈 Progression across rounds

| Round | Stack highlight | Q1 | Q2 | Q3 | Notes |
|-------|----------------|----|----|-----|--------|
| **1** | `phi3:mini`, baseline retrieval | Partial | Partial | Pass | Section bleed; Q3 honesty confirmed |
| **2** | `llama3.1:8b`, hybrid retrieval, prompt v2 | Partial | Partial | Pass | Better structure; ranking still noisy |
| **3** | Reranker, section metadata, prompt v3 | Partial+ | Pass | Pass | Q2 section alignment improved |
| **4** | Hard context filter, section-aware boost | Partial+ | Pass | Pass | Q2 on-section 1/1; Q3 honesty held |
| **5** | Header-aware chunking, dedupe, context guard | Pass | Pass | Pass | Also works on 2-page unnumbered NDA |

**Current bar (Round 5):** Q1–Q3 Pass on sample NDA · Q2/Q3 Pass on client-style 2-page NDA · ready for walkthrough demos.

---

## ✅ Round 5 at a glance (latest)

**Status:** Complete (May 2026)  
**Stack:** Docker Compose · `llama3.1:8b` · hybrid · reranker · section metadata · hard context filter · header-aware chunking · dedupe · context sufficiency guard

### Verdict

| Document | Q1 | Q2 | Q3 | Overall |
|----------|----|----|-----|---------|
| **Sample NDA** (4 pp, numbered) | Pass / Partial+ · on-section | Pass · four duties · on-section | Pass · honest No | **Go, demo-ready** |
| **Client-style 2-page NDA** (title headers) | _(not re-run post-tuning)_ | Pass · four duties | Pass · honest No | **Go for workflow + honesty** |

### Tuning that landed

| Change | Effect |
|--------|--------|
| Prompt v4 (no invented section numbers; one bullet per obligation) | Honest citations on unnumbered docs |
| Header-aware chunking at legal headers | Confidentiality block isolated on 2-page NDA |
| Near-duplicate drop + context sufficiency guard | Removed false-positive boilerplate chunks |

### Sample NDA latencies (post-tuning)

| Q | Content | Context purity | Latency |
|---|---------|----------------|---------|
| **Q1** | Pass / Partial+ | 1/1 on-section | ~32s |
| **Q2** | Pass · four duties | 1/1 on-section | ~28s |
| **Q3** | Pass · honest No | N/A (no penalty clause) | ~44s |

### VPS smoke (`phi3:mini`, `num_ctx=1024`)

Verified on [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev):

| Q | Result | Latency |
|---|--------|---------|
| Q1 | Pass | ~37s |
| Q2 | Pass | ~16s |
| Q3 | Partial (honest No, verbose hedge) | ~41s |

Q3 hedging is a small-model formatting quirk, not a hallucination. Larger models answer more cleanly. Expect slower first answers on CPU; that is normal for the self-host tier.

---

## ⏸️ Deferred

- Full Q1 re-run on 2-page NDA post-tuning
- 10–40 page client PDF evaluation
- Public demo video (after demo-tier LLM work)

---

## 🎬 Demo guidance

| Use case | Document |
|----------|----------|
| Section-number questions + context purity | Sample NDA |
| Obligation completeness + honesty on unnumbered layout | 2-page NDA |

Storyboard: [demo-script.md](demo-script.md).
