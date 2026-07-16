# Pilot evaluation — Round 4 plan

**Goal:** Fix **section-specific retrieval and answers** (Round 3’s main caveat) without regressing **Q3 honesty** or **client UX**.

**Builds on:** [Round 3 results](pilot-evaluation-round3.md) · [Round 3 plan](pilot-evaluation-round3-plan.md)

---

## Round 3 caveats → Round 4 targets

| Caveat | Symptom | Round 4 target |
|--------|---------|----------------|
| **Source misalignment** | Q1 Sources: WHEREAS, exclusions, title — not definition body | **100% on-section** in LLM context (post R4-E); optional dev metric: raw top-5 recall |
| **Wrong section in answer** | Q2 answered with **return/destroy on termination** (later sections) | Answer lists **four Section 3 duties** + material-breach sentence on **page 2** |
| **Adjacent-page bleed** | Page 2 mixes Section 3 + Section 4 in one chunk | Chunks ranked/filtered by **section**, not just page |
| **Honesty (must not regress)** | Q3 **pass** — no fake €/$ | Q3 stays **pass** after every experiment |
| **UX (done)** | Chat turns, Sources labels, ready state | **No further UI work** unless eval blocked |

**Not in Round 4:** VPS HTTPS (can run in parallel as **R4-E smoke**), client PDFs, persistent index (M9).

---

## Root-cause hypothesis

1. **Retrieval** returns lexically similar but **wrong-section** chunks (preamble, exclusions, termination).
2. **Chunking** (1000 chars / 400 overlap) **spans section boundaries** on a 4-page NDA.
3. **No section metadata** on chunks — hybrid search cannot prefer “Section 3” passages.
4. **Reranker off** — cross-encoder never re-orders fused candidates (`reranker.enabled: false` in config).
5. **Prompt** says “use only that section” but **context still contains** other sections → model follows strongest matching paragraph (return/destroy).

Round 4 tests these in **isolated experiments** so we know what actually moved the score.

---

## Experiment matrix (run in order)

Each step: same NDA, fresh session, **Q1 → Q2 → Q3**, record in `docs/pilot-evaluation-round4.md`.

| ID | Change | Config / code | Primary metric |
|----|--------|---------------|----------------|
| **R4-A** | **Reranker ON** | `rag.retrieval.reranker.enabled: true` | Source alignment Q1, Q2 |
| **R4-B** | **Section metadata at index** | Parse `^\d+\.` / `Section N` headers when chunking; store `metadata.section` | Q2 content + sources |
| **R4-C** | **Section-aware retrieval** | If query matches `Section N`, boost/filter candidates with `section == N` | Q1 + Q2 context **100% on-section** |
| **R4-D** | **MMR / page dedupe** | Prefer diverse pages/sections in top 5 (avoid three× page 1) | Q1 source diversity |
| **R4-E** | **Prompt v3 + context trim** | Hard rule: when section in question, **drop non-matching chunks** before LLM | Q2 wrong-section regression |
| **R4-F** | **Optional model A/B** | Same retrieval as best of A–E; compare `llama3.1:8b` vs legal-tuned model on Q2 only | Q2 content pass? |

**Stop rule:** Ship the **smallest combo** that hits success criteria (below). Do not stack all toggles without measuring each step.

---

## Proposed implementation steps

### Step 0 — Baseline instrumentation (½ day)

Before changing retrieval behaviour:

1. Add a **dev-only** “source checklist” after each answer (or in eval spreadsheet):
   - For each of 5 sources: page, first ~80 chars, **on-section Y/N**.
2. For Q2 failures, save **Exact Context fed to LLM** (already in dev expander) to confirm retrieval vs generation blame.
3. Create empty **`docs/pilot-evaluation-round4.md`** with the same scoring table as Round 3.

**Done when:** One documented Q2 run shows whether wrong answer text appears **in retrieved context**.

---

### Step 1 — R4-A: Enable reranker (config-only)

**File:** `configs/config.yaml`

```yaml
rag:
  retrieval:
    reranker:
      enabled: true   # was false
      top_n: 50
```

**Compose / local:** rebuild app container (reranker model loads on first query — allow +30–60s once).

**Re-run:** Q1, Q2, Q3.

**Expected:** Modest rank improvement; may **not alone** fix Q2 if chunks lack section tags.

**Rollback:** `enabled: false` if latency unacceptable on target VPS CPU.

---

### Step 2 — R4-B: Section metadata at chunk time

**Owner:** `rag-core-engineer` · **Files:** `src/app.py` (chunking path), optionally `src/rag/sectioning.py`

1. When splitting `page_docs`, detect section headers in chunk text:
   - Patterns: `^\s*\d+\.\s+`, `Section\s+\d+`, `ARTICLE\s+[IVX]+` (NDA uses numbered clauses).
2. Attach `metadata["section"] = "3"` (string) to each `Document` chunk — inherit from nearest header above chunk start.
3. Optionally attach `metadata["section_title"]` snippet for UI later.

**Unit check:** After indexing NDA, assert some chunks have `section == "1"` and `section == "3"`.

---

### Step 3 — R4-C: Section-aware retrieval

**Files:** `src/app.py` — `_hybrid_rrf_retrieval`, `_semantic_retrieval`, or new `_extract_target_section(query) -> str | None`

1. Parse target section from query: e.g. `Section 3`, `section 3`, `§3`.
2. After fusion (and rerank if on):
   - **Boost:** multiply score ×1.5 when `chunk.metadata.get("section") == target`.
   - **Or filter:** keep top candidates where section matches OR section unknown (preamble), cap at `top_k`.
3. If **no section** in query (Q3), behaviour unchanged.

**Re-run:** Q1, Q2, Q3.

**Success signal:** Q2 Sources include **hold in strict confidence**, **restrict access**, **limited use**, **return on request** language from Section 3.

---

### Step 4 — R4-D: MMR / dedupe (if page 1 still dominates)

**Options (pick one):**

- LangChain `MaxMarginalRelevanceRetriever` on fused pool before rerank.
- Simple dedupe: max **2 chunks per page** in top 5.
- Lower `chunk_overlap` for short contracts (e.g. 400 → 200) — **separate config experiment**, document in eval.

**Re-run:** Q1 only first (page-1 clutter worst on Q1).

---

### Step 5 — R4-E: Prompt v3 + hard context filter

**Files:** `configs/prompts.yaml`, retrieval assembly in `src/app.py`

1. **Prompt:** “If the question names Section N, use **only** chunks from Section N. If context contains other sections, ignore them.”
2. **Code (stronger):** When `target_section` set, `_assemble_context()` includes **only** matching chunks (fallback: warn + best-effort if &lt;2 chunks).

**Re-run:** Q2 — must not pull termination/return as **primary** obligations.

**Risk:** Over-filtering empty context → test Q3 still gets enough context to refuse amounts.

---

### Step 6 — R4-F: Optional model A/B (only if retrieval fixed)

Only after **Q2 context is 100% on-section** and content **Pass** (R4-E complete):

```bash
# Baseline
OLLAMA_MODEL=llama3.1:8b

# Optional
docker compose exec ollama ollama pull <law_model_tag>
OLLAMA_MODEL=<law_model_tag>
```

Compare **Q2 content only** — do not chase fluency if retrieval is still wrong.

---

### Step 7 — R4-E smoke: VPS (parallel track)

Not a retrieval experiment — validates **client demo path**:

1. Compose + Caddy + basic auth on VPS.
2. `APP_ALLOW_DEV_TOGGLE=false`, `APP_PRESENTATION_MODE=client`.
3. One full Q1–Q3 walkthrough on HTTPS; note cold latency.

Document in Round 4 eval under **Infrastructure**, not content scores.

---

## Evaluation protocol (unchanged questions)

1. How is **Confidential Information** defined in **Section 1**?
2. What **obligations** does **Section 3** impose on the Receiving Party?
3. Does this agreement specify **liquidated damages** or a **fixed financial penalty** for breach of confidentiality?

**Optional Q4 (positive control):** *What is the governing law and venue?* — expects Section 7 / page 4; confirms section routing works for lookup, not only honesty.

### Scoring

| Dimension | Pass (Round 4 bar) |
|-----------|-------------------|
| **Content** | Q1 partial→high partial; **Q2 lists four duties**; Q3 pass |
| **Context purity** (Q1, Q2) | **100% on-section** — every chunk in **Exact Context** and **Sources** is on-section (e.g. **2/2**, **1/1**). **Pass.** |
| **Context substance** (Q1, Q2) | **≥1** content-valid section chunk; **≥2** when the doc has distinct passages (4-page NDA may only yield 1–2) |
| **Format** | Pass (already stable) |
| **Latency** | Record; reranker +8B acceptable if &lt;2m warm on CPU |
| **Honesty** | Q3 **must pass** every experiment |

**Bar history:** R4-A–C used **≥3/5 on-section in raw top-5** (retrieval ranking quality). After **R4-E** hard context filter, **Sources** shows post-filter chunks only — **2/2** or **1/1** is a **pass**, not a fail. Optional **dev-only** metric: raw pre-filter top-5 on-section count (for reranker tuning, not go/no-go).

Record: model, hybrid, reranker on/off, section metadata on/off, hard filter on/off, wall times.

---

## Success criteria (Round 4 complete)

| # | Criterion |
|---|-----------|
| 1 | **Q2 content:** four obligations + material-breach line — **Pass / high partial** vs gold standard |
| 2 | **Q1 + Q2 context:** **100% on-section** in fed context (checklist Yes on every shown source) |
| 3 | **Q3:** **Pass** — no invented penalty amounts |
| 4 | Results written in **`docs/pilot-evaluation-round4.md`** |
| 5 | **Go / no-go** paragraph: ready for **client redacted PDF pilot** Y/N |

**No-go:** If Q2 still wrong-section after R4-A–E, document as **chunking / structure limit on sample NDA** and proceed to **client PDF** eval rather than more NDA tuning.

---

## Suggested timeline

```text
Day 1   Step 0 instrumentation + R4-A reranker eval
Day 2   R4-B section metadata + R4-C section retrieval (one PR)
Day 3   Re-run Q1–Q3; R4-D if needed
Day 4   R4-E prompt + context filter; full eval → pilot-evaluation-round4.md
Day 5   R4-F optional model A/B; VPS smoke (parallel)
```

---

## Deliverables

| Artifact | Path |
|----------|------|
| Round 4 observed results | `docs/pilot-evaluation-round4.md` (after eval) |
| Section chunking + retrieval | `src/app.py` (+ small helper if needed) |
| Config defaults (if shipped) | `configs/config.yaml`, `configs/prompts.yaml` |
| Tests | `tests/test_section_parsing.py` (minimal — section regex + metadata) |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| Reranker slow on CPU | Measure once; cache model; toggle via config; VPS sizing note in DEPLOYMENT |
| Section regex fails on client PDFs | Best-effort boost; fallback to hybrid-only; document in eval |
| Over-filter empties context | Minimum 2 chunks fallback; show retrieval warning in dev |
| Q3 regression after filter | Run Q3 after **every** change; honesty is non-negotiable |
| Over-tuning one NDA | Round 4 cap: **5 experiments** then client PDF decision |

---

## After Round 4

| Outcome | Next milestone |
|---------|----------------|
| **Go** on criteria | Client redacted PDF pilot + M7-7 demo recording on VPS |
| **Partial** (Q1/Q2 sources OK, Q2 prose weak) | Prompt + model A/B on client doc, not more NDA |
| **No-go** on section routing | Document limits; sell pilot as **honesty + workflow + sources for human review**, not auto counsel |

---

*Round 4 is retrieval-and-accuracy focused. UX and client presentation are treated as done after Round 3.*
