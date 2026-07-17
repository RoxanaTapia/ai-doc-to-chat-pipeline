# Documentation

## Background

Start here when you want to understand the product, deploy it, or contribute. Docs are split by audience so you can skip what you do not need.

> **Takeaway:** Clients start in `product/`. Operators and contributors start in `operators/`.

---

## 📖 Where to go

| If you are… | Read first | Then |
|-------------|------------|------|
| **Evaluating the product** | [How it works](product/architecture.md) | [Demo storyboard](product/demo-script.md), [Pilot evaluation](product/pilot-evaluation.md) |
| **Deploying on a VPS** | [DEPLOYMENT.md](../DEPLOYMENT.md) | [Architecture](product/architecture.md) |
| **Browsing as a buyer** | [README](../README.md) | Live pilot + [sample NDA](product/sample-nda.pdf) / [sample policy](product/sample-policy.md) |
| **Shipping milestones** | [PROJECT-DIRECTION](operators/PROJECT-DIRECTION.md) | [ROADMAP](operators/ROADMAP.md), [AGENTS.md](../AGENTS.md) |
| **Finding files in the repo** | [REPO-STRUCTURE](operators/REPO-STRUCTURE.md) | Target layout after M7.96 |
| **Testing OCR** | [Testing OCR](operators/testing-ocr.md) | Local Docker steps |

```mermaid
flowchart LR
  A[docs/README] --> B[product/]
  A --> C[operators/]
  A --> D[archive/]
  B --> E[Buyers / IT]
  C --> F[Contributors]
  D --> G[Historical eval]
```

---

## 🗂️ Folder map

```text
docs/
  README.md          ← you are here
  product/           ← how it works, demo, evaluation, sample docs
  operators/         ← roadmap, direction, repo structure, OCR testing
  archive/           ← older eval rounds (not primary reading)
```

| Folder | Role |
|--------|------|
| [`product/`](product/) | Client-readable: architecture, demo script, evaluation, sample NDA + policy |
| [`operators/`](operators/) | Roadmap, project direction, [repo structure](operators/REPO-STRUCTURE.md), contributor how-tos |
| [`archive/`](archive/) | Historical eval rounds; keep for reference, not day-one reading |


---

## 🔗 Quick links

- **Live pilot:** [ai-doc-pilot.roxanatapia.dev](https://ai-doc-pilot.roxanatapia.dev)
- **Public UI demo:** [Streamlit Cloud](https://ai-doc-to-chat-demo.streamlit.app)
- **Self-host:** [DEPLOYMENT.md](../DEPLOYMENT.md)
- **Sample uploads:** [product/sample-nda.pdf](product/sample-nda.pdf) · [product/sample-policy.md](product/sample-policy.md) (export to PDF)
