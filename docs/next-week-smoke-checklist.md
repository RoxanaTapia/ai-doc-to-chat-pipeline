# Next-Week Ollama Smoke Checklist

Use this as an operator runbook when the hardware arrives.

## 1) Pre-flight

- [ ] Activate project venv.
- [ ] Install deps (`pip install -r requirements.txt`).
- [ ] Confirm app runs in dummy mode first (`USE_DUMMY_GENERATOR=true`).
- [ ] Keep at least 1-2 realistic contract PDFs ready.

## 2) Model choice (CPU-first order)

- [ ] Try `phi3.5:mini` first (if available): `ollama pull phi3.5:mini`
- [ ] Fallback baseline: `ollama pull phi3:mini`
- [ ] Optional quality-heavy fallback: `llama3.1:8b` (quantized)

Set one model for the run:

- `configs/config.yaml` -> `rag.generation.model`, or
- `.env` -> `OLLAMA_MODEL=...`

## 3) Start local runtime

- [ ] Start server: `ollama serve`
- [ ] Verify model list: `ollama list`
- [ ] Confirm selected model appears.

## 4) Switch app to real generation

- [ ] Set `USE_DUMMY_GENERATOR=false` (or uncheck sidebar toggle).
- [ ] Optional for strict behavior: `OLLAMA_FALLBACK_TO_DUMMY=false`
- [ ] Launch app: `streamlit run src/app.py`

## 5) Smoke questions (run 3-5)

Use realistic legal prompts:

- "Summarize clause X."
- "Is there a non-compete longer than 2 years?"
- "Extract parties and effective dates."
- "What termination triggers are listed?"
- "What penalties are defined and where?"

## 6) Acceptance targets

- [ ] **Grounding:** Answers match retrieved chunks/pages, no obvious hallucinations.
- [ ] **Latency:** Average response under ~8-12s on CPU.
- [ ] **Stability:** No OOM or crashes on a 50-page contract.
- [ ] **Context:** Retrieval context around 12k-16k chars performs reliably.

## 7) Capture quick evidence

Record this in your test note:

- Model used:
- `top_k` and generation settings (`max_new_tokens`, `temperature`, `top_p`, `num_ctx`):
- Average latency over 3-5 prompts:
- Any timeout/OOM/errors:
- Pass/Fail for each acceptance target:

## 8) If failures occur

- Connection errors -> verify `ollama serve` is still running.
- Model not found -> `ollama pull <model>` and retry.
- Timeouts -> reduce context size / `top_k` / `max_new_tokens`.
- Weak grounding -> lower temperature, keep `do_sample=false`, refine prompt/chunks.
