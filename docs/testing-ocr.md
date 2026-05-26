# Testing OCR

Two levels: **unit tests** (fast, no binary required) and **integration test** (local Docker stack, real scanned PDF).

---

## Unit tests — run anywhere, no Tesseract needed

All external OCR dependencies are mocked. The tests cover `src/ocr.py` directly.

```bash
# One-time venv setup (skip if already done)
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pillow

# Run
pytest tests/test_ocr.py -v
```

Expected output: **13 passed**.

What is covered:

| Test | What it checks |
|------|----------------|
| `test_empty_string_is_scanned` | Empty text → scanned-page detection returns `True` |
| `test_text_at_boundary_is_not_scanned` | 50+ chars → not treated as scanned |
| `test_returns_stripped_text_and_no_error_on_success` | Happy path: pytesseract returns text → `(text, None)` |
| `test_returns_error_message_when_pytesseract_import_fails` | Missing dep → `("", "OCR dependencies not installed …")` |
| `test_returns_error_message_on_runtime_error` | Tesseract binary absent → `("", "OCR failed on page: …")` |
| `test_returns_error_message_on_os_error` | Corrupt pixmap → graceful error tuple |
| `test_returns_error_message_on_value_error` | Unsupported image mode → graceful error tuple |

---

## Integration test — local Docker stack with a real scanned PDF

### Prerequisites

- Docker and Docker Compose installed
- Tesseract is already in the Docker image (`apt-get install tesseract-ocr` — see `Dockerfile`)
- A **genuinely scanned PDF** (see below for how to create one)

### Create a scanned test PDF (if you don't have one)

On iPhone: photograph any printed page → open in Files → tap `…` → **Create PDF** → AirDrop to Mac.

Alternatively, on Mac: open any image in Preview → **File → Export as PDF**. The resulting PDF has image pages with no selectable text, which is what triggers the OCR path.

Do **not** use `docs/sample-nda.pdf` for this test — it is a digital PDF with extractable text and OCR will not trigger on it.

### Steps

1. **Configure `.env` for developer mode** (so the OCR diagnostics sidebar is visible):

   ```bash
   cp .env.example .env
   # Add these two lines to .env:
   APP_ALLOW_DEV_TOGGLE=true
   APP_PRESENTATION_MODE=developer
   USE_DUMMY_GENERATOR=false
   ```

2. **Start the local stack** (no Caddy needed):

   ```bash
   docker compose up -d ollama
   docker compose ps ollama   # wait for STATUS = healthy (~60 s)
   docker compose exec ollama ollama pull phi3:mini
   docker compose up --build -d app
   ```

3. **Open the app** at [http://localhost:8501](http://localhost:8501).

4. **Enable developer mode** in the sidebar (the toggle appears because `APP_ALLOW_DEV_TOGGLE=true`).

5. **Verify OCR is on**: sidebar → Advanced Options → "Enable OCR" toggle should be checked.

6. **Upload your scanned PDF.**

7. **Check for the OCR indicators:**
   - Orange warning: `"OCR used on N page(s). Results may vary by scan quality."`
   - Developer caption: `"OCR diagnostics: scanned pages detected=N, OCR applied=N, OCR unresolved=0"`

8. **Ask a question** about content on the scanned page. If Tesseract extracted text, you should get a grounded answer with source citations.

### What a passing integration test looks like

```
✅ Scanned PDF uploaded and indexed without crash
✅ OCR warning banner appears ("OCR used on N page(s)")
✅ Developer diagnostics show scanned_pages_detected > 0, OCR applied > 0
✅ Answer to a question about scanned content cites the correct page
```

### Failure modes and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| No OCR warning, no answer | PDF has selectable text — OCR not triggered | Use a true image-based PDF (see above) |
| `"OCR dependencies not installed"` | pytesseract or pillow missing in image | Rebuild: `docker compose build --no-cache app` |
| `"OCR failed on page: …"` | Tesseract binary absent | Check `Dockerfile` has `tesseract-ocr` in `apt-get install` |
| Answer is gibberish | Very low scan quality | Try a cleaner scan; OCR quality depends on DPI and contrast |

---

## Streamlit Cloud — OCR is not available there

Streamlit Cloud does not allow custom system packages like the `tesseract` binary. The `ocr_page_text` function handles this gracefully — it catches the `ImportError` and returns a user-facing warning. The feature is available only in the Docker stack (local or VPS).
