# Pilot evaluation — Round 5

**Status:** **Complete** — post–tuning #2/#3 regression · sample NDA + client-style 2-page NDA · May 2026  
**Plan (operator):** `docs-private/evaluation-round5-here-nda.md` (not in git)  
**Prior:** [Round 4](pilot-evaluation-round4.md) go on sample NDA

Round 5 tests **generalization** beyond the 4-page sample NDA: a **client-style 2-page personal NDA** with **title-style headers** (`Confidentiality.`, `Purpose.`) instead of numbered `Section N` clauses. After Round 5 baseline exposed a Q2 retrieval gap, **tuning #2** (header-aware chunking) and **tuning #3** (dedupe + context sufficiency guard) were applied and re-run on both documents.

---

## Short verdict — post–tuning #2/#3

| Document | Q1 definition | Q2 obligations | Q3 liquidated damages | Overall |
|----------|---------------|----------------|------------------------|---------|
| **Sample NDA** (4 pp) | **Pass / Partial+** · 1/1 on-section | **Pass** · four duties · 1/1 on-section | **Pass** · honest No | **Go** — demo-ready |
| **Client-style 2-page NDA** | _(not re-run post-tuning)_ | **Pass** · four duties incl. copy/pass-on | **Pass** · honest No | **Go** for workflow + honesty demo |

**Stack:** Docker Compose · `llama3.1:8b` · hybrid · `top_k: 5` · reranker **ON** · section metadata **ON** · hard context filter **ON** · `chunk_overlap: 200` · `split_on_legal_headers: true` · dedupe **ON** · context sufficiency guard **ON**

**Regression:** Sample NDA Q1–Q3 **pass** after header split + dedupe (Round 4 bar held).

---

## Tuning shipped (Round 5 follow-up)

| ID | Change | Primary effect |
|----|--------|----------------|
| **#1** | Prompt v4 — no invented section numbers; one bullet per obligation line | Honest citations on unnumbered docs; complete duty lists |
| **#2** | Header-aware chunking — split at `N.` and `Title.` legal headers before character split | HERE **Confidentiality** block isolated (~2k chars); obligation language in context |
| **#3** | Near-duplicate drop + context sufficiency guard + dev sidebar defaults | Sample Q3: no lone “Agreement.” at rank 1; skip LLM when context lacks duty markers |

**Dev UX:** Indexing logs (chunk count + FAISS build) persist in **document status above chat** so they do not appear below a “Thinking…” turn.

---

## Sample NDA — regression re-run (14 chunks · header split ON)

| Q | Content | Context purity | Latency (observed) |
|---|---------|----------------|--------------------|
| **Q1** — Section 1 definition | **Pass / Partial+** — definition + examples | **1/1 on-section** · ~498 chars · 5 chunks fed | **31.9s** (retrieval 2.1s · gen 29.8s) |
| **Q2** — Section 3 obligations | **Pass** — four duties incl. return/destroy | **1/1 on-section** · ~600 chars | **27.9s** (retrieval 1.4s · gen 26.4s) |
| **Q3** — Liquidated damages | **Pass** — No; injunctive relief only; no invented €/$ | N/A (no section in query) · ~2514 chars · 4 chunks | **44.1s** (retrieval 1.3s · gen 42.8s) |

**Notes:** Q3 sources still mix oral-info / misc / irreparable-harm passages (expected — no penalty clause exists). Answer stayed honest. Dedupe removed boilerplate “Agreement.” false-positive rank.

---

## Client-style 2-page NDA — post–tuning #2/#3 (10 chunks · header split ON)

| Q | Content | Sources / context | Latency (observed) |
|---|---------|-------------------|--------------------|
| **Q2** — What must signatory **not do** with CI? | **Pass** — disclose without consent; restrict use; no public announcement; no discuss/copy/pass-on | Source 1 = **Confidentiality** block (rel 1.0) · ~1699 chars · 2 chunks with obligation language | **~25–32s** warm |
| **Q3** — Liquidated damages? | **Pass** — clear **No**; no fixed amount invented | Irrelevant sources OK (no penalty clause in doc) | **~32s** warm |

**Baseline (pre–tuning #2):** Q2 **failed** — obligation bullets not in Exact Context; only generic “keep confidential” in answer. Post–header split, **Confidentiality** paragraph retrieved and all core duties listed.

**Demo guidance:** Use **sample NDA** for section-number questions (Q1/Q2 + context purity story). Use **2-page NDA** for obligation completeness and **Q3 honesty** on unnumbered layouts.

---

## Round 5 vs Round 4 bar

| Metric | Round 4 (sample only) | Round 5 post-tuning |
|--------|----------------------|---------------------|
| Sample Q2 on-section | 1/1 | **1/1** (held) |
| Sample Q3 honesty | Pass | **Pass** (held) |
| Unnumbered-doc Q2 | N/A | **Pass** (was fail pre–#2) |
| Unnumbered-doc Q3 | N/A | **Pass** |

---

## Deferred

- Full Q1 re-run on 2-page NDA post–tuning (content was Partial+ pre-tuning; sources misaligned — lower priority than Q2 fix).
- Longer multi-page client PDF eval (10–40 pp).
- VPS smoke / M7-7 demo recording — warm-up query first; hero answers on sample NDA.

---

## Related docs

- [Round 4 results](pilot-evaluation-round4.md) · [Round 4 plan](pilot-evaluation-round4-plan.md)
- Operator protocol + gold answers: `docs-private/evaluation-round5-here-nda.md`
