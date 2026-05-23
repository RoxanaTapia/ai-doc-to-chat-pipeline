# Pilot evaluation — Round 4

**Status:** Baseline + **R4-A recorded** · sample NDA (4 pages) · May 2026  
**Plan:** [Round 4 plan](pilot-evaluation-round4-plan.md) · **Prior:** [Round 3](pilot-evaluation-round3.md)

Round 4 focuses on **section-specific retrieval and answers** (Round 3’s main caveat) without regressing **Q3 honesty** or **client UX**.

**Instrumentation (Step 0):** In developer mode, each assistant answer shows:

- **📋 Source checklist (eval)** — page, ~80 char excerpt, auto on-section Y/N (confirm manually).
- **📄 Exact Context fed to LLM** — persisted per answer for Q2 retrieval vs generation diagnosis.

---

## Short verdict

| Question | Content (R4-A) | Format | Sources (R4-A) | Overall |
|----------|----------------|--------|----------------|---------|
| **Q1** — Section 1 definition | **Partial+** (~90%) | **Pass** | **Fail bar** (2/5 manual) | Answer improved; sources unchanged vs baseline |
| **Q2** — Section 3 obligations | **Pass** | **Pass** | **Fail bar** (1/5 manual) | **Sec 3 now rank 1**; four other sources still off-section |
| **Q3** — Liquidated damages | **Pass** | **Pass** | **Partial** (noise OK) | Honesty held |
| **Q4** _(optional)_ — Governing law / venue | _Not run_ | — | — | |

**Stack (R4-A):** Docker Compose · `llama3.1:8b` · hybrid · `top_k: 5` · page separators ON · reranker **ON** (`BAAI/bge-reranker-base`, top_n 50) · section filter **off**  
**Latency (R4-A):** Q1 **3m 7s** (retrieval **1m 34s** cold · gen 1m 33s) · Q2 **1m 0s** (retrieval 1.4s · gen 58.9s) · Q3 **56.7s** (retrieval 1.2s · gen 55.5s)

**R4-A verdict:** **Partial success** — reranker improves **Q2 lead chunk** and keeps Q3 honest, but **≥3/5 on-section sources** still not met on Q1/Q2. **Next: R4-B + R4-C** (section metadata + section-aware retrieval).

---

## Baseline vs R4-A (at a glance)

| Metric | Baseline (reranker off) | R4-A (reranker on) |
|--------|-------------------------|---------------------|
| Q1 sources (manual /5) | 2/5 | **2/5** (no change) |
| Q2 sources (manual /5) | 1/5 | **1/5** (count same; **rank 1 = Sec 3**) |
| Q2 Sec 3 chunk rank | 4 (score 0.628) | **1 (score 1.0)** |
| Q3 honesty | Pass | **Pass** |
| Q1 retrieval time | 0.1s | **1m 34s** (cold reranker load) |
| Q2+ retrieval time | ~0.1s | **~1.2–1.4s** (warm) |

---

## Step 0 — Baseline (pre-retrieval changes)

| Run | Q2 wrong-section text in **Exact Context**? | Auto on-section Q1 | Auto on-section Q2 | Notes |
|-----|---------------------------------------------|--------------------|--------------------|-------|
| **Baseline** | **Yes** — Page 3 termination/destroy alongside Page 2 Sec 3 | **2/5** | **1/5** | Sec 3 duties **in** context but drowned by noise; generation OK this run |

**Q2 diagnostic (baseline):** Context contains **both** Section 3 duties **and** termination/destroy language. Retrieval primarily at fault for source alignment; Sec 3 at rank 4 still allowed a good answer.

---

## Experiment log

| ID | Change | Q1 content | Q1 sources (on-section /5) | Q2 content | Q2 sources (/5) | Q3 pass? | Notes |
|----|--------|------------|----------------------------|------------|-----------------|----------|-------|
| **Baseline** | Step 0 instrumentation only | Partial | **2/5** | Pass / high partial | **1/5** | **Yes** | Hybrid; reranker off |
| **R4-A** | Reranker ON | Partial+ | **2/5** | **Pass** | **1/5** | **Yes** | Sec 3 → rank **1**; Q1 retrieval cold **~94s**; sources bar still failed |
| **R4-B** | Section metadata at index | | | | | | _Next_ |
| **R4-C** | Section-aware retrieval | | | | | | _Next (with R4-B)_ |
| **R4-D** | MMR / page dedupe | | | | | | If Q1 page clutter persists |
| **R4-E** | Prompt v3 + context trim | | | | | | After retrieval fixed |
| **R4-F** | Model A/B (optional) | | | | | | Q2 only, if needed |

**Config snapshot per run:** model · hybrid · reranker on/off · section metadata on/off · section filter on/off · wall times.

**R4-A ops notes:** First query after container start pays reranker model download + CrossEncoder init on CPU (~90s in retrieval). Q2/Q3 warm retrieval ~1s. Dev expanders without per-message keys can stay expanded across turns — UX fix deferred.

---

## R4-A — Cross-encoder reranker ON

### Q1 — How is **Confidential Information** defined in **Section 1**?

**Observed answer:**

- **Summary / Details:** Commercial value + labeled “Confidential”; adds **“reasonable person would understand its confidential nature”**; example list (business plans, technical data, customer lists, etc.).

**Source checklist (manual confirm):**

| # | Page | On-section | Excerpt (~80 chars) | Rank / score |
|---|------|------------|---------------------|--------------|
| 1 | 1 | **N** | WHEREAS… parties agree as follows: 1.… | 1 · 1.0 |
| 2 | 1 | **N** | CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT — Acme… | 2 · 0.408 |
| 3 | 1 | **N** | 2. Exclusions from Confidential Information… | 3 · 0.092 |
| 4 | 3 | **N** | 5. Treatment of Oral Information… | 4 · 0.01 |
| 5 | 4 | **N** | 7. Miscellaneous — laws of Spain… | 5 · 0.005 |

Auto checklist: **2/5** (heuristic tags preamble/title as Y — confirm **N** manually).

**Exact Context:** Page 1 — WHEREAS + Section 1 definition + Section 2 exclusions (overlapping chunks). Definition reaches LLM; **source list** still preamble-heavy.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Partial+** (~90%) | Slightly richer than baseline |
| Format | **Pass** | |
| Sources | **Fail bar** | **2/5** manual (bar ≥3/5) — **no improvement vs baseline** |
| Latency | **3m 7s** | retrieval **1m 34s** (cold reranker) · generation 1m 33s |

---

### Q2 — What **obligations** does **Section 3** impose on the Receiving Party?

**Observed answer:**

- **Summary / Details:** Hold in strict confidence; restrict access (need-to-know + signed NDAs); use only for evaluating potential business relationship; return or destroy on written request.

**Exact Context diagnostic:** Page 2 leads with **“3. Obligations of the Receiving Party”** (+ Section 4 duration bleed). Page 3 still has termination/destroy. **Sec 3 now rank 1 in sources and context** — major reranker win vs baseline (rank 4).

**Source checklist (manual confirm):**

| # | Page | On-section | Excerpt (~80 chars) | Rank / score |
|---|------|------------|---------------------|--------------|
| 1 | 2 | **Y** | 3. Obligations of the Receiving Party — hold… restrict… | **1 · 1.0** |
| 2 | 3 | **N** | destroy all physical and electronic copies… | 2 · 0.158 |
| 3 | 1 | **N** | CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT… | 3 · 0.146 |
| 4 | 3 | **N** | 5. Treatment of Oral Information… | 4 · 0.107 |
| 5 | 4 | **N** | 7. Miscellaneous — laws of Spain… | 5 · 0.099 |

Auto checklist: **1/5**.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Four duties — stable vs baseline |
| Format | **Pass** | |
| Sources | **Fail bar** | **1/5** manual (bar ≥3/5); **ranking fixed**, diversity not |
| Latency | **1m 0s** | retrieval 1.4s · generation 58.9s |

---

### Q3 — Does this agreement specify **liquidated damages** or a **fixed financial penalty**?

**Observed answer:**

- **Summary:** No liquidated damages or fixed financial penalty.
- **Details:** Material breach (Section 3); injunctive relief and damages (Sections 5–6) — **no €/$/% invented.**

**Source checklist:** No section in question — on-section **N/A**.

**Sources (top ranks):** Page 2 breach/duration (1.0) · Page 3 injunctive (0.836) · Page 2 Sec 3 (0.617) · Page 3 oral (0.342) · Page 3 records (0.336).

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Honesty test met — **no regression** |
| Format | **Pass** | |
| Sources | **Partial** | Noisy; sufficient for refusal |
| Latency | **56.7s** | retrieval 1.2s · generation 55.5s |

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
| Content vs doc | — | Section 7 / page 4 positive control — run after R4-C |

---

## Infrastructure (parallel track)

| Check | Result |
|-------|--------|
| VPS HTTPS + basic auth smoke | _TBD_ |
| Client presentation mode on VPS | _TBD_ |
| Cold vs warm latency on VPS | _TBD_ — expect reranker cold hit on first query |

---

## Go / no-go (Round 4 complete)

| # | Criterion | Met? |
|---|-----------|------|
| 1 | Q2 lists four obligations + material-breach line | **Yes** (baseline + R4-A) |
| 2 | Q1 + Q2 sources **≥3/5 on-section** | **No** (2/5 and 1/5 in R4-A) |
| 3 | Q3 **Pass** — no invented penalty amounts | **Yes** (R4-A) |
| 4 | Results recorded in this doc | **Yes** (baseline + R4-A) |
| 5 | Ready for **client redacted PDF pilot** | **No** — need R4-B/C |

**Decision:** **No-go for client PDF pilot.** R4-A kept — reranker helps Q2 ranking. **Proceed R4-B + R4-C** as one implementation pass, then re-run Q1–Q3 (+ optional Q4).

---

*Round 4 is retrieval-and-accuracy focused. UX and client presentation are treated as done after Round 3.*
