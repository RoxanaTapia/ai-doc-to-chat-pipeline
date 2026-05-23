# Pilot evaluation — Round 3

**Status:** Local Docker · sample NDA (4 pages) · May 2026  
**Prior rounds:** [Round 1](pilot-evaluation-round1.md) · [Round 2](pilot-evaluation-round2.md) · [Plan](pilot-evaluation-round3-plan.md)

Round 3 combines **full eval re-run** (same three questions, same stack as Round 2) with **client-demo UX fixes** discovered during the session — chat turn flow, Sources labeling, ready-state copy, and Streamlit error chrome.

Retrieval tuning (reranker, section filter) from the Round 3 plan is **not yet applied** — this round records baseline eval **plus UX ship**.

---

## Short verdict

| Question | Content | Format | Sources | Overall |
|----------|---------|--------|---------|---------|
| **Q1** — Section 1 definition | **Partial** (~85%) | **Pass** | **Partial** | Good Markdown; excerpts still mix preamble / exclusions |
| **Q2** — Section 3 obligations | **Fail / Partial** | **Pass** | **Partial** | Answer described **return/destroy on termination** — not Section 3 duties |
| **Q3** — Liquidated damages | **Pass** | **Pass** | **Partial** | No invented €/$; qualitative damages only |

**Stack:** Docker Compose · **`llama3.1:8b`** · hybrid · `top_k: 5` · page separators ON · prompt v2  
**Latency:** Q1 **1m 44s** (cold) · Q2 **59.6s** · Q3 **1m 7s** · retrieval sub-second

**Buyer takeaway:** Demo **UX is client-ready** after Round 3 fixes. **Content accuracy on section-specific questions** (especially Q2) and **source alignment** remain the honest limits — next work is retrieval precision, not more UI polish on the same NDA.

---

## UX fixes shipped (Round 3)

| Issue | Fix |
|-------|-----|
| User question invisible until AI finished | Show **user bubble immediately**; **Thinking…** in assistant bubble below |
| `Sources · answer 2` felt opaque | **Sources — {truncated question}** e.g. *Sources — What obligations does Section 3…* |
| `6,895 characters extracted` on client view | Client: **N pages processed** only; full stats in developer mode |
| Expander `key=` crash on Streamlit 1.54 | Removed unsupported key; stable titles from question preview |
| Ask Google / ChatGPT on errors | `showErrorLinks = false` in `.streamlit/config.toml` |
| Upload lost on rerun (boot UUID) | Process-stable session boot (`pid`) |

---

## Questions & observed results

### Q1 — How is **Confidential Information** defined in **Section 1**?

**Observed answer:**

- **Summary / Details** — commercial value, labeled “Confidential”, non-exhaustive list (business plans, technical data, customer lists, etc.).
- Structure scannable; no redundant ### Source block in body (prompt v2).

**Sources expander:**

- Source 1: WHEREAS preamble (page 1)
- Source 2: Section 2 exclusions (page 1)
- Source 3: Agreement title / parties (page 1)
- Sources 4–5: page 2 obligation / duration adjacent text

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Partial** | Core definition present; “reasonable person” not seen |
| Format | **Pass** | Summary + Details · UX **5/5** |
| Sources | **Partial** | Readable excerpts (**4/5**) · on-section alignment **2/5** |
| Latency | 1m 44s | Cold 8B on CPU |

---

### Q2 — What **obligations** does **Section 3** impose on the Receiving Party?

**Observed answer:**

- **Summary** focused on **return or destroy** copies and **written certification** after termination — language from **later sections**, not the four Section 3 duties (hold confidential, restrict access, limited use, return on request).
- **Material breach** / four-obligation framing **missing** from the written answer.

**Sources expander:** pages 1–3 mix — exclusions, WHEREAS, termination / injunctive relief — **not Section 3-centric**.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Fail / Partial** | Wrong section theme — **regression vs Round 1/2 expectation** |
| Format | **Pass** | Markdown structure OK |
| Sources | **Partial** | No clear Section 3 duty excerpts in top 5 |
| Latency | 59.6s | Warm |
| Action | **Round 4 retrieval** — section filter + reranker before client PDFs |

---

### Q3 — Does this agreement specify **liquidated damages** or a **fixed financial penalty**?

**Observed answer:**

- **Summary:** Correct — no liquidated damages or fixed financial penalty.
- **Details:** No specific amount; references **injunctive relief and damages** (Section 6 / page 3 in output).
- **No €/$/% invented.**

**Sources expander:** page 3 injunctive relief relevant; pages 1–2 noise.

| Dimension | Score | Notes |
|-----------|-------|-------|
| Content vs doc | **Pass** | Honesty test met |
| vs gold standard | **Meets bar** | Minor section-number variance |
| Format | **Pass** | |
| Sources | **Partial** | Key page 3 present; top ranks still mixed |
| Latency | 1m 7s | |
| UX | Sources label | *Sources — Does this agreement specify liquidated…* — clear for buyers |

---

## Round 3 vs Round 2

| Area | Round 2 | Round 3 |
|------|---------|---------|
| Chat UX | User message delayed; generic “answer N” labels | **Immediate user turn**; **question-scoped Sources** |
| Q1 content | ~85% partial | ~85% partial (stable) |
| Q2 content | Expected ~85% partial (not fully captured) | **Wrong section** — needs retrieval work |
| Q3 honesty | Pass | **Pass** (stable) |
| Client-ready caption | Character count shown | **Pages only** |

---

## Next steps (from plan)

1. Enable **cross-encoder reranker** + **section boost** — re-run Q1–Q3 on same NDA.  
2. Record results in **Round 4** or append to this doc.  
3. **HTTPS VPS** smoke test with `APP_ALLOW_DEV_TOGGLE=false`.  
4. **Go / no-go** for client redacted PDFs after source alignment ≥3/5 on Q1–Q2.

Questions or a pilot on your PDFs: [README → For teams and consulting](../README.md#for-teams-and-consulting).
