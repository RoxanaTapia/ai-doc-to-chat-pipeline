# Pilot evaluation — Round 1

**Status:** Local Docker pilot · sample confidentiality agreement (4 pages) · May 2026

This document records an **honest evaluation** of the reference stack on a realistic document type — not a marketing scorecard. It shows what the pilot does well today, where answers need human review, and how we test before a client engagement.

Architecture and deployment: [architecture-pilot.md](architecture-pilot.md) · [DEPLOYMENT.md](../DEPLOYMENT.md)

---

## Short verdict

| Question | Result vs document | vs gold standard | Notes |
|----------|-------------------|------------------|--------|
| **Q1** — Section 1 definition | **Partial** | Content ~80%; format partial | Core definition correct; blended Sections 2 & 5; missed “reasonable person” clause |
| **Q2** — Section 3 obligations | **Partial** | Content ~85%; format partial | Four duties + key sentence on page 2; one bullet from Section 4 (adjacent retrieval noise) |
| **Q3** — Spanish statutes | **Pass** | **Meets gold standard** | Correct refusal; no invented law; governing-law clause only |

**Stack:** Docker Compose · `phi3:mini` · first answer **~87s** (CPU cold start); retrieval **~0.1s**.

**Gold-standard answers** in this doc are the **evaluation bar**, not guaranteed output today. **Q3-style honesty is already there**; **Q1–Q2** need source checks and tighter questions — improvable with prompt, retrieval, and a stronger model without changing the bar.

**Buyer takeaway:** Grounded Q&A with visible sources works for evaluation; strict section-only answers are a **tuning** target, not a reason to skip a pilot on your PDFs.

---

## What we tested

| Item | Detail |
|------|--------|
| **Document** | Sample confidentiality / NDA-style agreement (4 pages, digital PDF) |
| **Environment** | Docker Compose on laptop — app + Ollama (`phi3:mini`) |
| **Goal** | Confirm grounded Q&A, visible sources, and honest refusals before VPS or client demos |

Three question types mirror how we run pilots:

1. **Definition** — how a key term is defined in the contract  
2. **Section lookup** — what a numbered section says about a specific topic (with sources)  
3. **Honesty** — ask for detail the agreement does **not** contain  

---

## Results summary

| Test | Outcome | Client takeaway |
|------|---------|-----------------|
| **Definition** | Found core definition plus related clauses from nearby sections | Useful for exploration; **verify section boundaries** on critical terms |
| **Section obligations** | Retrieved the right page and cited the key sentence; answer mixed one nearby section | **Sources panel** lets you spot this — why we show page + excerpt |
| **Outside the document** | Correctly refused to name specific statutes; cited governing-law clause only | Strong **honesty** signal — no invented legal detail |

**First-answer latency (CPU):** ~70–90 seconds on cold start — normal for this stack; subsequent questions are faster.

---

## What worked

- **Real generation** with **page-level sources** (similarity score, page number, excerpt in the UI)
- **Fast retrieval** (~0.1s) — generation time dominates on CPU
- **Definition and obligation content** largely matched the document when the question targeted a clear section
- **Refusal** when asked for Spanish regulations not named in the agreement — only “governed by the laws of Spain” appears in the text

---

## What we learned (current limits)

These are **expected** for an evaluation pilot, not bugs to hide:

| Observation | Implication |
|-------------|-------------|
| Definition answers can **blend** exclusions or oral-disclosure clauses from other sections | Ask section-specific questions for due diligence; use sources to validate |
| Section questions may **pull adjacent paragraphs** (e.g. duration text near obligations) | Shows why evaluation includes **your** PDFs before production tuning |
| Contract asks about “material breaches in Section 3” but the text defines **obligations** and states any breach of them is material | **Question wording matters** — we use “What obligations does Section 3 impose?” in demos |
| No **calculator** — answers are read from retrieved text, not computed | Invoice totals printed on the page: OK; line-item math: not guaranteed |

See also [Pilot scope & limitations](../README.md#pilot-scope--limitations) in the README.

---

## Demo questions used (Round 1)

**Revised set** (aligned with document structure after Round 1):

1. How is **Confidential Information** defined in **Section 1**?
2. What **obligations** does **Section 3** impose on the Receiving Party?
Question 3 (Round 1) is intentionally **not answerable** from the document alone — it tests that the system does not hallucinate legal citations.

---

## Demo questions — Round 2

Same document (`NDA_contract.pdf`). **Q1 and Q2 unchanged.** **Q3 replaced** so Round 2 tests a different honesty trap (models often invent dollar amounts).

1. How is **Confidential Information** defined in **Section 1**?
2. What **obligations** does **Section 3** impose on the Receiving Party?
3. Does this agreement specify **liquidated damages** or a **fixed financial penalty** for breach of confidentiality?

Question 3 (Round 2) is **not answerable with a number** from the text — the agreement mentions **material breach** and, in places, **damages** or **injunctive relief**, but **no fixed euro/dollar amount**.

---

## Expected answer format (gold standard)

Buyers should know **what “good” looks like** before a pilot on their own PDFs. We score Round 1 against the format below — not against generic ChatGPT fluency.

### Format we expect in every answer

| Element | Why it matters |
|---------|----------------|
| **Grounded in the document** | No outside legal knowledge presented as if it were in the contract |
| **Section + page reference** | Lets counsel or ops verify in seconds |
| **Sources panel aligns with the answer** | Excerpts on the cited pages support the claims |
| **Clear scope** | Definitions stay in definition sections; exclusions cited separately if asked |
| **Honest gaps** | If the text does not contain the detail, say so — do not invent statutes, totals, or clauses |

**Ideal UI shape:** a concise answer (bullets or short paragraphs) **plus** expandable sources showing page number, similarity score, and excerpt.

---

### Question 1 — Section 1 definition

**Question:** How is **Confidential Information** defined in **Section 1**?

**Expected answer (example):**

> Under **Section 1** (page 1), **Confidential Information** means any information disclosed by the **Disclosing Party** that has **commercial value** and is labeled **“Confidential”**, including (non-exhaustive): business plans, technical data, customer lists, financial projections, product roadmaps, pricing strategies, and marketing materials.
>
> The same section also states that information is deemed confidential if it is **marked as such** or if a **reasonable person** would understand its confidential nature.
>
> *(Section 1 defines the term; exclusions appear separately in Section 2 and oral-disclosure rules in Section 5 — not part of the core definition unless the question asks for them.)*

**Sources you would expect:** page **1**, excerpts containing “Definition of Confidential Information” and “reasonable person”.

**Round 1 observed:** Core definition largely correct; answer also pulled **Section 2 exclusions** and **Section 5 oral / 10-day** rules into a single “definition” reply, and omitted the **reasonable person** clause. Acceptable for exploration; **not** ideal for strict contract review without reading sources.

---

### Question 2 — Section 3 obligations

**Question:** What **obligations** does **Section 3** impose on the Receiving Party?

**Expected answer (example):**

> **Section 3 — Obligations of the Receiving Party** (page 2) requires the Receiving Party to:
>
> 1. **Hold** all Confidential Information in **strict confidence**
> 2. **Restrict access** to employees, contractors, and third parties with a legitimate need to know who are bound by **identical nondisclosure restrictions**
> 3. **Not use** the information for any purpose other than **evaluating a potential business relationship**
> 4. **Return or destroy** all materials containing Confidential Information upon **written request** of the Disclosing Party
>
> The section further provides that **any breach of these obligations shall constitute a material breach** of the Agreement. It does **not** list separate named “material breaches” — it defines duties and treats violation of those duties as material.
>
> *(Do not attribute Section 4 duration or “shorter term” language to Section 3.)*

**Sources you would expect:** page **2**, especially an excerpt containing *“Any breach of these obligations shall constitute a material breach”*.

**Round 1 observed:** The four obligations were largely correct and **Source 3 (page 2)** hit the key sentence. The written answer also included **Section 4** language about asserting a shorter confidentiality period — visible in sources as adjacent noise on page 2. This is exactly why we ship **sources** and run **Round 1 on your PDFs** before production.

---

### Question 3 (Round 1) — Honesty / statutes not in the document

**Question:** Does this agreement identify any **specific Spanish statutes** governing confidentiality?

**Expected answer (example):**

> **No.** The agreement does **not** name any specific Spanish statutes, codes, or regulations that govern confidentiality.
>
> **Section 7 — Miscellaneous** (page 4) states only that the Agreement **shall be governed by the laws of Spain** and that disputes shall be resolved in the **courts of Madrid**. It does not identify particular legal provisions (for example, specific data-protection or civil-code articles).
>
> I cannot answer which statutes apply beyond what the contract text states without external legal research — that would go outside this document.

**Sources you would expect:** page **4**, excerpt with *“governed by the laws of Spain”* and *“courts of Madrid”*.

**Round 1 observed:** **Met gold standard** — correct refusal, no invented regulations, governing-law clause acknowledged. Strongest signal for buyers who fear hallucinated compliance claims.

---

### Question 3 (Round 2) — Honesty / no fixed penalty in the document

**Question:** Does this agreement specify **liquidated damages** or a **fixed financial penalty** for breach of confidentiality?

**Expected answer (example):**

> **No.** This agreement does **not** set a **liquidated damages** clause or a **fixed euro/dollar penalty** for breach of confidentiality.
>
> **Section 3** (page 2) states that **any breach of its obligations shall constitute a material breach** — but does not quantify damages. **Section 5** (page 3) mentions **injunctive relief and damages** for breach of the oral-information protocol, again **without a stated amount**. There is no schedule of fines, caps, or per-day penalties in the text.
>
> I cannot state a specific financial penalty from this document; that would require inventing numbers not present in the agreement.

**Sources you might see (supporting “no fixed amount”):** page **2** (material breach language); page **3** (qualitative “damages” only). **No source should contain a numeric penalty.**

**Why this question:** Legal chatbots often hallucinate **“€X per day”** or **“10% of contract value”** — a strong Round 2 honesty test, distinct from Round 1’s statute question.

**Round 2 observed:** **Pass** on honesty (no invented penalty); see full write-up in [pilot-evaluation-round2.md](pilot-evaluation-round2.md).

---

### Why this matters for your purchase decision

| If you need… | Round 1 tells you… |
|--------------|-------------------|
| **Fast first-pass review** of policies and contracts with **citations** | The workflow works; question wording and source checks are part of the process |
| **Replace counsel or guarantee legal conclusions** | This is **not** that product — evaluation pilot only |
| **Zero hallucination risk** | Q3-style honesty is a design goal; Q1–Q2 still require **human verification** via sources |
| **Production on your corpus** | Round 1 on a **sample** NDA → Round 2+ on **your** redacted PDFs under NDA, then tuning and persistence |

A serious buyer should ask: *“Can we run the same three question types on our documents and compare answers to our own gold standard?”* — that is the intended pilot engagement.

---

## System progress at this stage

| Area | Round 1 status |
|------|----------------|
| PDF upload + chunking + FAISS retrieval | Working |
| Hybrid retrieval (semantic + BM25) | Available |
| Source previews in chat | Working |
| Local Ollama generation | Working |
| HTTPS + basic auth deployment | Documented ([DEPLOYMENT.md](../DEPLOYMENT.md)); VPS validation next |
| Persistent document library | Planned ([ROADMAP.md](ROADMAP.md) M9) |
| API / agents / SSO | Planned (M8–M10) |

**Next evaluation steps:** **Round 3** complete — see [Round 3 results](pilot-evaluation-round3.md); retrieval tuning + VPS next.

---

## Round 2 — configuration (May 2026)

Applied in repo for the next evaluation pass on `NDA_contract.pdf`:

| Change | Setting |
|--------|---------|
| Retrieval breadth | `top_k: 5` (was 10) |
| Retrieval strategy | **Hybrid** (semantic + BM25) default |
| Context labels | **Page separators on** by default |
| Prompt | Section-scoped, Markdown sections (Summary / Details / Source) |
| Answer length | `max_new_tokens: 384` |
| Sources UI | **5** sources max; ~280-char excerpts at word/sentence boundaries |
| Q3 (honesty) | Liquidated damages / fixed penalty — **no amount in doc** |

**Model (manual):** pull and set `OLLAMA_MODEL=llama3.1:8b` for Round 2 if RAM allows (~8 GB+); keep `phi3:mini` for CPU-only comparison.

**Re-test checklist:** same Q1–Q2; new Q3 above; score vs gold standard; see [Round 2 results](pilot-evaluation-round2.md) and [Round 3 plan](pilot-evaluation-round3-plan.md).

---

## How to read this as a buyer

- This pilot is for **evaluation** — grounded search + chat with **inspectable sources**, on **your** infrastructure.  
- Round 1 proves the **workflow** (upload → ask → cite → refuse when appropriate), not perfect accuracy on every question type.  
- Production engagements add tuning on **your** documents, persistence, auth, and ops — scoped after a successful pilot.

Questions or a pilot on your PDFs: [README → For teams and consulting](../README.md#for-teams-and-consulting).
