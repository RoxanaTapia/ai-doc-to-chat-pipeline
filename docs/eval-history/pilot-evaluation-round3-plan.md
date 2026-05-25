# Pilot evaluation — Round 3 plan

**Goal:** Close the gap between **good answers** and **trustworthy sources** — the main Round 2 finding — then validate on **HTTPS VPS** before client PDFs.

**Builds on:** [Round 1](pilot-evaluation-round1.md) · [Round 2 results](pilot-evaluation-round2.md) · **[Round 3 observed (UX + eval)](pilot-evaluation-round3.md)**

---

## Round 2 carry-forward (problem statement)

| What worked | What still hurts |
|-------------|------------------|
| Markdown **Summary / Details** | Retrieved excerpts often **wrong section** (Q1: exclusions / WHEREAS vs definition) |
| **Honesty** on Q3 (no fake €/$) | **Source ranking** does not match cited section |
| Hybrid + `top_k: 5` | Q2 **page 2 adjacency** noise (Section 4 bleed) |
| Sources expander UX | Buyers need **section ↔ excerpt** alignment, not just pretty text |

Round 3 optimizes **retrieval precision** and **demo reliability**, not new features for their own sake.

---

## Proposed scope

### 1. Retrieval & ranking (priority)

| Experiment | Config / code | Success metric |
|------------|---------------|----------------|
| **Section filter** | Parse “Section N” from question; boost or filter chunks whose metadata / text lead with that section | Q1 sources ≥3/5 from Section 1 body; Q2 ≥3/5 from Section 3 |
| **Reranker ON** | `rag.retrieval.reranker.enabled: true` (BGE cross-encoder) | Source alignment +1 vs Round 2 on same three questions |
| **MMR diversity** | Optional `fetch_k` + MMR to reduce duplicate page-1 chunks | Fewer repeated page-1 excerpts in top 5 |
| **`top_k` A/B** | 4 vs 5 vs 6 on 4-page NDA | Best F1 on manual source checklist |

**Definition of done:** For each demo question, a reviewer can open **Sources** and see **≥3 of 5** excerpts clearly supporting the answer section (binary checklist in eval doc).

---

### 2. Generation (secondary)

| Experiment | Notes |
|------------|--------|
| **`llama3.1:8b` baseline** | Keep as default (Round 2) |
| **`initium/law_model` or Saul** | Optional A/B on same 3 questions — measure section fidelity, not fluency |
| **Prompt v3** | Explicit: “In Details, cite section + page inline once; do not duplicate Sources panel.” |
| **`max_new_tokens`** | 384 vs 512 if Details truncate obligations lists |

---

### 3. UX & demo polish

| Item | Owner | Notes |
|------|-------|-------|
| Sources expander only | Done post-R2 | No ### Source block in answers |
| Stable chat turns | Done post-R2 | No ghost excerpts between questions |
| **Section tag in source header** | Streamlit | e.g. **Source 2 · Section 3 · Page 2** (parse from chunk / page map) |
| **Collapsible history** | Optional | Long demos: collapse older turns |
| **Ready / indexing states** | Done | Green banner before chat |
| Client mode default | Done | No dev toggle on Compose |

---

### 4. Infrastructure

| Step | Purpose |
|------|---------|
| Deploy **Compose + Caddy** on VPS | Buyer-facing HTTPS URL |
| Basic auth | Protect demo |
| **`APP_ALLOW_DEV_TOGGLE=false`** | Client-only sidebar |
| Log **model id + retrieval mode** per answer (dev) | Reproducible eval notes |

---

## Round 3 question set

**Same three types** (continuity across rounds):

1. How is **Confidential Information** defined in **Section 1**?
2. What **obligations** does **Section 3** impose on the Receiving Party?
3. Does this agreement specify **liquidated damages** or a **fixed financial penalty** for breach of confidentiality?

**Optional Q4 (Round 3 add-on):** *What is the governing law and venue?* — positive lookup (Section 7, page 4) to balance honesty-only Q3.

---

## Scoring template (repeat each round)

For each question, score **1–5** (subjective) plus **Pass / Partial / Fail** vs gold standard:

| Dimension | 1 | 3 | 5 |
|-----------|---|---|---|
| **Content** | Wrong / hallucination | Mostly right, section bleed | Matches gold standard |
| **Format** | Wall of text | Markdown, minor issues | Summary + Details scannable |
| **Sources** | Irrelevant excerpts | Mixed relevance | ≥3/5 on-section excerpts |
| **Latency** | >2m CPU | ~1m | <45s warm |

Record **model**, **retrieval mode**, **reranker on/off**, and **wall time** in the eval table.

---

## Suggested execution order

```text
Week A — Retrieval
  ├─ Enable reranker + re-run 3 questions
  ├─ Section boost/filter prototype
  └─ Update Round 3 observed in eval doc

Week B — Model & prompt
  ├─ Optional law_model A/B vs llama3.1:8b
  └─ Prompt v3 if Source inline citations still duplicate UI

Week C — VPS demo
  ├─ HTTPS + auth smoke test
  ├─ One recorded walkthrough (M7-7)
  └─ README / sales one-pager pointer to live URL
```

---

## Out of scope for Round 3

- Persistent index (M9) — note as roadmap, don’t block eval
- Multi-PDF library
- API / SSO
- Committing client NDA PDF to public repo

---

## Deliverables

| Artifact | Location |
|----------|----------|
| Round 3 observed results | [pilot-evaluation-round3.md](pilot-evaluation-round3.md) |
| Round 4+ (retrieval tuning) | Append after reranker / section filter |
| Config deltas | `configs/config.yaml`, `configs/prompts.yaml` |
| VPS checklist | `DEPLOYMENT.md` |
| Demo script | `docs/demo-script.md` (M7-7) |

---

## Success criteria (Round 3 complete)

1. **Q1–Q3** scored vs gold standard with **Q3 pass** and **Q1–Q2 partial → high partial** on source alignment.  
2. **HTTPS demo** reachable with client UI (no dev toggle).  
3. **No chat UI regressions** (ghost sources, lost uploads, unclear ready state).  
4. Written **go / no-go** for “pilot on client redacted PDFs” in eval doc.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| 8B too slow on VPS CPU | Offer `phi3:mini` env override for latency demos; 8B for quality demos |
| Section parser brittle on non-standard PDFs | Fallback to hybrid only; section boost as best-effort |
| Reranker RAM / latency | Toggle via config; measure on target VPS size |
| Over-tuning on one NDA | Round 3 ends with **client PDF checklist**, not more NDA tweaks |

---

*Round 3 is the last eval pass on the sample NDA before shifting tuning to client documents under NDA.*
