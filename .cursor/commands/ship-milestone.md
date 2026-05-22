# Ship or plan a milestone (M7–M12)

Act as **milestone-orchestrator** for milestone: **$ARGUMENTS** (e.g. M7, M8).

## If planning only (no code)

- Read `docs/ROADMAP.md` for that milestone.
- List GitHub issues (create draft issue bodies if none exist for M8+).
- Produce: issue order, dependencies, agent assignments, parallel windows, human blockers, expected PR count, 2-week schedule.
- Invoke **blocker-reporter** for undecided items without AGENTS.md defaults.
- **Do not write code.**

## If implementing

- Work **one issue at a time** unless I explicitly request `/parallel-m7-bootstrap`.
- Follow `/ship-m7-issue` workflow per issue for M7 (#33–#39).
- For M8+: dispatch **rag-core-engineer**, **config-guardian**, **deploy-engineer** as ROADMAP specifies.

## Milestone definitions of done

- **M7:** HTTPS demo, DEPLOYMENT.md, 7 PRs, real Ollama
- **M8:** FastAPI `/health`, `/chat`, OpenAPI
- **M9:** Postgres/pgvector persistence survives reboot
- **M10:** SSO + SECURITY.md
- **M11:** RUNBOOK.md + monitoring
- **M12:** Anthropic provider + SERVICES.md + demo video link

Report: schedule, blockers, recommended next issue for human to approve.
