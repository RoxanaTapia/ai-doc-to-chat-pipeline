# Ship or plan a milestone (M7.8–M12)

Act as **milestone-orchestrator** for milestone: **$ARGUMENTS** (e.g. M7.8, M8, M9).

## If planning only (no code)

- Read `docs/ROADMAP.md` and `docs/PROJECT-DIRECTION.md` for that milestone.
- List GitHub issues; draft bodies for any missing work.
- Produce: issue order, dependencies, agent assignments, parallel windows, human blockers, expected PR count.
- Invoke **blocker-reporter** for undecided items without AGENTS.md defaults.
- **Do not write code.**

## If implementing

- Work **one issue at a time** in a **new Agent chat** per issue.
- Use `/ship-issue #NN` workflow.
- PR bodies must start with **`## Main contribution`** (see milestone-workflow rule).

## Milestone definitions of done

- **M7:** ✅ HTTPS pilot, DEPLOYMENT.md, Compose — shipped
- **M7.8:** Swappable LLM, Anthropic demo tier, streaming, recordable walkthrough
- **Video (#57):** Link in README after M7.8
- **Portfolio packaging:** Calm README hero, pilot + Cloud links, 16:9 thumbnail story (after video)
- **M8 (thin):** `/health`, `/chat`, OpenAPI; modest `src/rag/` extract — not a large rewrite gate
- **M8.5:** Eval export — optional / secondary after thin M8
- **M9–M11:** Client-triggered (persist, SSO, runbook) — not default portfolio path
- **M12:** Light services/tiers one-pager (providers ship in M7.8)

Follow portfolio order in `docs/ROADMAP.md`. Do not plan Support MVP / n8n CRM as this repo’s M8+.

Report: schedule, blockers, recommended next issue for human to approve.
