# Ship or plan a milestone train (M7.8–M12)

Act as **milestone-orchestrator** for milestone: **$ARGUMENTS** (e.g. M7.8, M8, delivery).

Follow **train mode** in `AGENTS.md` unless the operator said `hold merges`. Keep prose calm; avoid salesy framing.

## If planning only (no code)

- Read `docs/operators/ROADMAP.md` and `docs/operators/PROJECT-DIRECTION.md` for that milestone.
- List GitHub issues; draft bodies for any missing work.
- Produce: issue order, dependencies, agent assignments, parallel windows, human blockers, expected PR count.
- Invoke **blocker-reporter** for undecided items without AGENTS.md defaults.
- **Do not write code.**

## If implementing (train mode)

- Run the queue **one issue at a time** in this conductor chat when possible.
- Still **one issue = one branch = one PR**.
- Use the `/ship-issue #NN` loop for each issue: specialists → verifier → commit → PR → merge when green → **status pulse**.
- Respect parallel windows (e.g. #55 ∥ #56) and serial deps (#53→#54→#55; #58→#59→#60).
- On hard gates (#57 video URL, secrets, CI still red): Blocker card (`docs-writer` polish) and **STOP**.
- Do not start Support MVP / n8n CRM as this repo’s next work.

### Default delivery train

```text
#53 → #54 → (#55 ∥ #56) → #57 → packaging → #58 → #59 → #60
```

End with a phase-complete pulse after packaging + thin M8 (or after packaging if M8 is deferred).

## Milestone definitions of done

- **M7:** ✅ HTTPS pilot, DEPLOYMENT.md, Compose (shipped)
- **M7.8:** Swappable LLM, Anthropic demo tier, streaming, recordable walkthrough
- **Video (#57):** Link in README after M7.8
- **Portfolio packaging:** Calm README hero, pilot + Cloud links, 16:9 thumbnail story (after video)
- **M8 (thin):** `/health`, `/chat`, OpenAPI; modest `src/rag/` extract (not a large rewrite gate)
- **M8.5:** Eval export (optional / secondary after thin M8)
- **M9–M11:** Client-triggered (persist, SSO, runbook); not default portfolio path
- **M12:** Light services/tiers one-pager (providers ship in M7.8)

Follow portfolio order in `docs/operators/ROADMAP.md`.

Report after each wave: pulse (or blocker), and the next issue.
