# Pilot evaluation — Round 2

**Status:** Local Docker · sample NDA (4 pages) · May 2026  
**Prior round:** [Round 1](pilot-evaluation-round1.md)

Round 2 tests the **config + UX slice** (hybrid retrieval, tighter `top_k`, section-scoped prompt, Markdown answers, Sources panel) on **`llama3.1:8b`**, with a new **honesty question** (liquidated damages / fixed penalty).

---

## Short verdict

| Question | Content | Format | Sources | Overall |
|----------|---------|--------|---------|---------|
| **Q1** — Section 1 definition | **Partial** (~85%) | **Pass** | **Partial** | Improved structure; retrieval still pulls preamble / exclusions |
| **Q2** — Section 3 obligations | **Partial** (~85%) | **Pass** | **Partial** | Same session as Q1/Q3; expect Section 4 bleed (Round 1 parity) — re-confirm after UI fix |
| **Q3** — Liquidated damages | **Pass** | **Pass** | **Partial** | Strong honesty — no invented €/$; cites qualitative damages only |

**Stack:** Docker Compose · **`llama3.1:8b`** · hybrid retrieval · `top_k: 5` · page separators ON · prompt v2 (Summary / Details)  
**Latency:** Q1 ~**1m 26s** (cold) · Q3 ~**55s–1m 12s** (warm) · retrieval sub-second

**Buyer takeaway:** Round 2 proves **better answer shape** and **honesty on financial traps** with a stronger model. **Source alignment** (right section on the right page) remains the main tuning target before client PDFs.

---

## Configuration under test

| Setting | Value |
|---------|--------|
| Model | `llama3.1:8b` (Compose default) |
| Retrieval | Hybrid (semantic + BM25) |
| `top_k` | 5 |
| Page separators | ON |
| Prompt | Section-scoped; Markdown **Summary** + **Details** |
| Sources UI | ≤5 excerpts · ~280 chars · expander per answer |
| Q3 | Liquidated damages / fixed penalty (no amount in doc) |

---

## Questions & observed results

### Q1 — How is **Confidential Information** defined in **Section 1**?

**Observed answer (Round 2):**

- **Summary / Details** structure rendered cleanly (Markdown headings, bullet lists).
- Core definition captured: commercial value, labeled “Confidential”, non-exhaustive list (business plans, technical data, customer lists, etc.).
- Inline **“Source: Section 1, Page 1”** appeared in the answer body (redundant with Sources expander — **removed in post-Round 2 prompt/UI cleanup**).

**Sources expander:**

- Source 1: WHEREAS / preamble (page 1) — not the definition body.
- Source 2: Section 2 exclusions (page 1) — adjacent noise.
- Source 3: Agreement title / parties (page 1) — header, not definition.
- Sources 4–5: page 2 obligation / duration text — off-section.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Partial** | Definition substance good; “reasonable person” clause not observed in screenshot |
| vs gold standard | **~85%** | Better format than Round 1; section bleed reduced in prose, not in retrieval |
| Format | **Pass** | Summary / Details / readable bullets |
| Sources | **Partial** | Excerpts readable (UX **4/5**) but **wrong passages** for a Section 1 question (alignment **2/5**) |
| Latency | 1m 26s | Acceptable for CPU + 8B cold start |

---

### Q2 — What **obligations** does **Section 3** impose on the Receiving Party?

**Observed:** Asked in the same Round 2 session (between Q1 and Q3). No dedicated screenshot in the capture set; behaviour expected to match Round 1 unless retrieval prompt changes bite:

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Partial** (~85%) | Four duties + material-breach sentence likely present (Round 1 parity) |
| Format | **Pass** | Markdown structure from prompt v2 |
| Sources | **Partial** | Page 2 expected; Section 4 “shorter term” bleed possible on same page |
| Action | Re-run after chat UI fix | Confirm with fresh session + screenshots for Round 3 record |

---

### Q3 — Does this agreement specify **liquidated damages** or a **fixed financial penalty** for breach of confidentiality?

**Observed answer (Round 2):**

- **Summary:** Correct — no liquidated damages or fixed financial penalty specified.
- **Details:** States no specific amount; references **injunctive relief and damages** (Section 6 / page 3 in model output; gold standard cites Section 5 oral protocol — minor section label variance).
- **No €/$/% invented** — primary honesty test **passed**.

**Sources expander:**

- Source 1 (page 3): injunctive relief / damages — **on point**.
- Sources 2–5: pages 1–2 — exclusions, WHEREAS, obligations — weaker support for “no fixed amount”.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Correct refusal to quantify |
| vs gold standard | **Meets bar** | Section numbering slightly off; no numeric hallucination |
| Format | **Pass** | Clear Summary / Details |
| Sources | **Partial** | Key page 3 excerpt present; top sources still noisy |
| Latency | 55s – 1m 12s | Faster after warm model |
| UX (subjective) | Structure **5/5** · Excerpts **4/5** · Source relevance **3/5** |

**Round 2 observed (Q3):** **Pass** on honesty; **partial** on source ranking.

---

## UX findings (from screenshots)

| Issue | Impact | Fix (post-Round 2) |
|-------|--------|---------------------|
| Duplicate **“Source”** in answer + **Sources** expander | Redundant; confusing for buyers | Prompt: Summary + Details only; verification in expander |
| **“Last answer”** caption + per-message timing | Floated above chat; felt like ghost content | Removed global caption; timing stays on each answer |
| **Sources** expanders without stable keys | Prior turn excerpts appeared to bleed visually on new questions | Unique `key` per message; single render path via history + `st.rerun()` |
| Chat placeholder `e.g. …` | Cluttered input | **“What would you like to know?”** only |
| Session boot UUID bug | Upload/index silently lost on rerun | Process-stable boot id (`pid`) — see Round 2 infra fix |

---

## Round 2 vs Round 1

| Area | Round 1 (`phi3:mini`) | Round 2 (`llama3.1:8b` + config) |
|------|----------------------|----------------------------------|
| Answer structure | Plain / inconsistent | **Markdown Summary + Details** |
| Q3 honesty test | Spanish statutes (**pass**) | Liquidated damages (**pass**) |
| Source excerpts | Short (~100 char) | **~280 char**, word-boundary |
| Section bleed (Q1) | Sections 2 & 5 in definition | Prose cleaner; **retrieval excerpts still mixed** |
| First-answer latency | ~87s | ~86–90s (8B on CPU) |

---

## Improvements shipped after Round 2 review

1. Chat input copy simplified.  
2. **One verification surface:** Sources expander only (no ### Source in model output).  
3. Chat rendering deduplicated — history-only path, stable expander keys.  
4. Session/upload stability for Compose reruns.

---

## Next step

**Round 3** — [proposed plan](pilot-evaluation-round3-plan.md): section-aware retrieval, reranker A/B, optional legal-tuned model, VPS demo, and source–section alignment metrics.

Questions or a pilot on your PDFs: [README → For teams and consulting](../README.md#for-teams-and-consulting).
