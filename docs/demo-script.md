# 5-minute demo script (M7-7)

Use this script when recording a sales/portfolio video (Loom, OBS, or QuickTime).

**Tools:** OBS Studio or Loom (record); DaVinci Resolve (optional trim); Excalidraw/Canva (architecture slide).

---

## Storyboard

| Time | Scene |
|------|--------|
| 0:00–0:30 | Title: *Private RAG — your PDFs, cited answers, dedicated infrastructure* |
| 0:30–1:00 | Show architecture diagram (`docs/architecture-pilot.md` or slide) |
| 1:00–2:00 | Open pilot URL; upload a real PDF (contract or policy sample) |
| 2:00–2:30 | Indexing spinner (edit: jump-cut long waits) |
| 2:30–3:30 | Ask two questions — one general, one specific (e.g. notice period) |
| 3:30–4:15 | Expand **Sources** — page refs and similarity scores |
| 4:15–4:45 | Privacy line: *Runs on dedicated server; documents not sent to public ChatGPT* |
| 4:45–5:00 | CTA: pilot in client environment — contact / README link |

---

## Checklist before recording

- [ ] `USE_DUMMY_GENERATOR=false` on demo host
- [ ] Ollama model pulled (`llama3.1:8b` or `phi3:mini`)
- [ ] Sample PDF prepared (non-sensitive or redacted)
- [ ] Browser zoom 100%; close unrelated tabs
- [ ] Blur email/hostname in post if needed

---

## Hosting the video

- Unlisted YouTube, Loom link, or portfolio embed
- Add URL to README under *Reference deployment demo*
