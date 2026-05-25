# Pilot evaluation — Round 4

**Status:** **Complete** — baseline · R4-A · **R4-B/C/D/E post-fix recorded** · sample NDA (4 pages) · May 2026  
**Plan:** [Round 4 plan](pilot-evaluation-round4-plan.md) · **Prior:** [Round 3](pilot-evaluation-round3.md)

Round 4 focuses on **section-specific retrieval and answers** (Round 3’s main caveat) without regressing **Q3 honesty** or **client UX**.

**Instrumentation (Step 0):** In developer mode, each assistant answer shows:

- **📋 Source checklist (eval)** — page, ~80 char excerpt, auto on-section **Yes/No** (confirm manually).
- **📄 Exact Context fed to LLM** — persisted per answer for Q2 retrieval vs generation diagnosis.

**Source bar (updated after R4-E):** Pass = **100% on-section** in fed context (e.g. **2/2**, **1/1**). The old **≥3/5 raw top-5** rule applied before hard context filter; see [plan](pilot-evaluation-round4-plan.md#scoring).

---

## Short verdict — final stack (R4-B/C/D/E post-fix)

| Question | Content | Context purity | Overall |
|----------|---------|----------------|---------|
| **Q1** — Section 1 definition | **Pass / Partial+** | **2/2 Yes** (100%) | **Pass** |
| **Q2** — Section 3 obligations | **Pass** (four duties) | **1/1 Yes** (100%) | **Pass** |
| **Q3** — Liquidated damages | **Pass** | N/A (no section in query) | **Pass** |
| **Q4** _(optional)_ — Governing law / venue | _Not run_ | — | — |

**Stack (final):** Docker Compose · `llama3.1:8b` · hybrid · `top_k: 5` · page separators ON · reranker **ON** · section metadata **ON** · section-aware boost **ON** · hard context filter **ON** (content-aware + trim) · `chunk_overlap: 200` · `max_chunks_per_page: 2`  
**Latency (final):** Q1 **2m 34s** (retrieval **1m 35s** cold · gen 58.6s) · Q2 **18.9s** (retrieval 1.1s · gen 17.8s) · Q3 **56.9s** (retrieval 1.0s · gen 55.9s)

**Final verdict:** **Go** for **client redacted PDF pilot** on sample NDA criteria. Q2 Section 4 regression fixed; Q3 honesty held. Optional polish: Q1 near-duplicate sources, Q4 positive control, VPS smoke.

---

## Baseline vs R4-A vs final (at a glance)

| Metric | Baseline | R4-A | **Final (R4-B/C/D/E)** |
|--------|----------|------|------------------------|
| Q1 context on-section | Mixed (WHEREAS + Sec 1 + Sec 2) | Mixed | **100% (2/2)** |
| Q2 context on-section | Mixed (Sec 3 + Page 3 termination) | Mixed | **100% (1/1)** |
| Q2 Sec 3 in context | Yes (rank 4) | Yes (rank 1) | **Yes only** (no Sec 4 bleed) |
| Q2 content | Pass | Pass | **Pass** (four duties incl. return/destroy) |
| Q3 honesty | Pass | Pass | **Pass** |
| Q1 retrieval (warm/cold) | 0.1s | 1m 34s cold | 1m 35s cold (first query) |
| Q2+ retrieval | ~0.1s | ~1.2–1.4s | **~1.0–1.1s** |

---

## Step 0 — Baseline (pre-retrieval changes)

| Run | Q2 wrong-section text in **Exact Context**? | Auto on-section Q1 | Auto on-section Q2 | Notes |
|-----|---------------------------------------------|--------------------|--------------------|-------|
| **Baseline** | **Yes** — Page 3 termination/destroy alongside Page 2 Sec 3 | **2/5** | **1/5** | Sec 3 duties **in** context but drowned by noise; generation OK this run |

**Q2 diagnostic (baseline):** Context contains **both** Section 3 duties **and** termination/destroy language. Retrieval primarily at fault for source alignment; Sec 3 at rank 4 still allowed a good answer.

---

## Experiment log

| ID | Change | Q1 content | Q1 context purity | Q2 content | Q2 context purity | Q3 pass? | Notes |
|----|--------|------------|-------------------|------------|-------------------|----------|-------|
| **Baseline** | Step 0 instrumentation only | Partial | Mixed | Pass | Mixed | **Yes** | Hybrid; reranker off |
| **R4-A** | Reranker ON | Partial+ | Mixed | **Pass** | Mixed | **Yes** | Sec 3 → rank **1** in raw top-5 |
| **R4-B/C** | Section metadata + section-aware | — | — | — | — | — | Shipped with E post-fix |
| **R4-D** | Overlap 200 + max 2/page | — | — | — | — | — | Shipped with E post-fix |
| **R4-E** | Hard filter + prompt v3 + B.1/E.1 | Partial+ | **2/2** | **Pass** | **1/1** | **Yes** | Q2 Sec 4 regression **fixed** |
| **R4-F** | Model A/B (optional) | — | — | — | — | — | _Skipped — not needed on sample NDA_ |

**Config snapshot per run:** model · hybrid · reranker on/off · section metadata on/off · hard filter on/off · wall times.

**Ops notes:** First query after container start pays reranker model download + CrossEncoder init on CPU (~90s in retrieval). Re-index required after chunk_overlap / tagging changes (uploader ✕ reset + re-upload).

---

## R4 final — R4-B/C/D/E post-fix

### Q1 — How is **Confidential Information** defined in **Section 1**?

**Observed answer:**

- **Summary / Details:** Commercial value; labeled “Confidential”; reasonable-person standard; examples (business plans, technical data, customer lists, financial projections, etc.).

**Source checklist (auto):**

| # | Page | On-section | Excerpt (~80 chars) | Score |
|---|------|------------|---------------------|-------|
| 1 | 1 | **Yes** | 1. Definition of Confidential Information — commercial value… | 1.5 |
| 2 | 1 | **Yes** | 1. Definition of Confidential Information — (near-duplicate) | 0.576 |

Auto checklist: **2/2** (100% on-section).

**Exact Context:** 719 chars · **2 chunks** · Page 1 Section 1 definition only (no WHEREAS / exclusions in fed context).

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass / Partial+** | Definition + examples correct |
| Context purity | **Pass** | **2/2** on-section |
| Format | **Pass** | |
| Latency | **2m 34s** | retrieval **1m 35s** (cold reranker) · generation 58.6s |

**Minor nit:** Two Page 1 sources are near-duplicates — optional dedupe polish (R4-D.2).

---

### Q2 — What **obligations** does **Section 3** impose on the Receiving Party?

**Observed answer:**

- **Summary / Details:** Hold in strict confidence; restrict access (need-to-know + signed NDAs); use only for evaluating potential business relationship; **return or destroy** on written request.

**Source checklist (auto):**

| # | Page | On-section | Excerpt (~80 chars) | Score |
|---|------|------------|---------------------|-------|
| 1 | 2 | **Yes** | 3. Obligations of the Receiving Party — hold… restrict… | 1.5 |

Auto checklist: **1/1** (100% on-section).

**Exact Context diagnostic:** 611 chars · Section 3 duties only — **no Section 4 Duration bleed**. Return/destroy language **in context** and **in answer**. Material-breach sentence present in context.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Four duties — **Sec 4 regression fixed** vs pre-fix R4-E run |
| Context purity | **Pass** | **1/1** on-section |
| Format | **Pass** | |
| Latency | **18.9s** | retrieval 1.1s · generation 17.8s (warm) |

---

### Q3 — Does this agreement specify **liquidated damages** or a **fixed financial penalty**?

**Observed answer:**

- **Summary:** No liquidated damages or fixed financial penalty.
- **Details:** Injunctive relief and damages (Page 3) — **no €/$/% invented.**

**Source checklist:** No section in question — on-section **N/A**.

**Sources (top ranks):** Page 3 oral/injunctive (1.0) · Page 2 Sec 3 + Sec 4 (0.376) · Page 2 duration (0.215) · Page 3 oral confirm (0.207) · Page 4 misc (0.15).

**Exact Context:** 4170 chars · 5 chunks · mixed sections (filter off — no section in query).

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Honesty test met — **no regression** |
| Format | **Pass** | |
| Sources | **Pass** | Noisy OK for negative lookup |
| Latency | **56.9s** | retrieval 1.0s · generation 55.9s |

---

## R4-A — Cross-encoder reranker ON _(historical)_

<details>
<summary>R4-A detailed results — click to expand</summary>

**Stack:** reranker ON · section filter **off** · bar at time: **≥3/5 raw top-5** (superseded after R4-E).

### Q1 (R4-A)

- Content Partial+ (~90%); raw sources **2/5** on-section; latency **3m 7s**.
- Exact Context: WHEREAS + Section 1 + Section 2 exclusions.

### Q2 (R4-A)

- Content Pass (four duties); raw sources **1/5** on-section; Sec 3 at **rank 1** (score 1.0).
- Exact Context: Sec 3 lead + Sec 4 bleed + Page 3 termination noise.

### Q3 (R4-A)

- Pass; no invented amounts; latency **56.7s**.

**R4-A verdict (historical):** Partial success on old bar — reranker improved Q2 ranking; context still mixed until R4-E.

</details>

---

## Baseline — observed results (Step 0)

<details>
<summary>Baseline Q1–Q3 (reranker off) — click to expand</summary>

### Q1 baseline

- Content partial (~85%); sources **2/5** manual; latency **1m 34s** (retrieval 0.1s).
- Top sources: WHEREAS, Sec 2 exclusions, title, Page 2 wrong-section chunks.

### Q2 baseline

- Content pass/high partial; sources **1/5**; Sec 3 at **rank 4** (0.628); latency **48.6s**.
- Exact Context: Sec 3 + Page 3 termination noise.

### Q3 baseline

- Pass; no invented amounts; latency not fully captured (~similar warm stack).

</details>

---

### Q4 _(optional)_ — What is the **governing law and venue**?

**Observed answer:** _Not run._

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | — | Section 7 / page 4 positive control — run on client PDF or before demo recording |

---

## Infrastructure (parallel track)

| Check | Result |
|-------|--------|
| VPS HTTPS + basic auth smoke | _TBD_ |
| Client presentation mode on VPS | _TBD_ |
| Cold vs warm latency on VPS | _TBD_ — expect reranker cold hit on first query; warm-up query before demo |

---

## Go / no-go (Round 4 complete)

| # | Criterion | Met? |
|---|-----------|------|
| 1 | Q2 lists four obligations + material-breach line in context | **Yes** |
| 2 | Q1 + Q2 context **100% on-section** (2/2 and 1/1) | **Yes** |
| 3 | Q3 **Pass** — no invented penalty amounts | **Yes** |
| 4 | Results recorded in this doc | **Yes** |
| 5 | Ready for **client redacted PDF pilot** | **Yes** (sample NDA) |

**Decision:** **Go** for client redacted PDF pilot on sample NDA. Round 4 retrieval stack (R4-A through R4-E) **shipped**. Do not over-tune sample NDA further; validate on **client document structure** next. Optional: Q4, Q1 dedupe, VPS smoke (M7-6 / M7-7).

**Follow-up:** [Round 5](pilot-evaluation-round5.md) — client-style 2-page NDA + tuning #2/#3 regression on sample NDA.

---

*Round 4 is retrieval-and-accuracy focused. UX and client presentation are treated as done after Round 3.*
