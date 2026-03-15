# Next-Week Ollama Smoke Test Checklist

Operator runbook for validating full local generation once hardware arrives.

## 1. Pre-flight
- [ ] Activate project venv
- [ ] `pip install -r requirements.txt` (if needed)
- [ ] Confirm app works in dummy mode (`USE_DUMMY_GENERATOR=true`)
- [ ] Prepare 1–2 real contract PDFs

## 2. Model Selection (CPU priority)
Recommended order:
- [ ] `phi3.5:mini` -> `ollama pull phi3.5:mini` (preferred, newer/lighter)
- [ ] Fallback: `phi3:mini`
- [ ] Stronger option (more RAM): `llama3.1:8b` (quantized)

Set model in:
- `configs/config.yaml` -> `rag.generation.model`
- or `.env` -> `OLLAMA_MODEL=phi3.5:mini`

## 3. Start Ollama
- [ ] Run `ollama serve` (keep terminal open)
- [ ] `ollama list` -> confirm model is downloaded
- [ ] Optional quick test: `ollama run <model>`

## 4. Switch to Real Generation
- [ ] Set `USE_DUMMY_GENERATOR=false` (or uncheck sidebar toggle)
- [ ] Strict mode (optional): `OLLAMA_FALLBACK_TO_DUMMY=false`
- [ ] Launch: `streamlit run src/app.py`

## 5. Run Smoke Tests (3–5 realistic questions)
Good legal-style prompts:
- "Is there a non-compete clause longer than 2 years?"
- "Summarize clause X"
- "Extract parties and effective date"
- "What are the termination triggers?"
- "What penalties are defined and on which page?"

## 6. Acceptance Criteria
- [ ] Grounding: answers match retrieved chunks/pages, no clear hallucinations
- [ ] Latency: average ~8–12 seconds on CPU
- [ ] Stability: no OOM/crashes on 50-page PDF
- [ ] Context handling: ~12k–16k chars of retrieval works reliably

## 7. Quick Results Log
Note:
- Model used:
- Key settings (`top_k`, `max_new_tokens`, `temperature`, `num_ctx`):
- Average latency (3–5 prompts):
- Errors (if any):
- Pass/Fail per criterion

## 8. Troubleshooting
- Connection fail -> check `ollama serve` is running
- Model missing -> `ollama pull <model>`
- Timeouts -> reduce `top_k` / `max_new_tokens`
- Poor grounding -> set `temperature=0`, refine prompt/chunk overlap

Once this passes, full local generation is ready for real use.
