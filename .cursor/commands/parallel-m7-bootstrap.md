# Parallel M7 bootstrap (one-time)

Act as **milestone-orchestrator**. Run the **one-time** parallel bootstrap for M7 early issues.

**Prerequisite:** Confirm with me that no VPS/domain is required yet (placeholders OK).

## Parallel batch (Task tool — 3 subagents at once)

| Agent | Scope ONLY | Files |
|-------|------------|-------|
| deploy-engineer | M7-1 Dockerfile + .dockerignore | Dockerfile, .dockerignore |
| config-guardian | M7-4 .env.example self-host vars | .env.example |
| docs-writer | M7-5 DEPLOYMENT.md **skeleton** | DEPLOYMENT.md or docs/ (placeholders) |

**Do NOT** let any agent touch files outside their column.

## After parallel batch returns

1. List file conflicts (if any).
2. List blockers via **blocker-reporter** if needed.
3. Recommend **separate PRs** merge order: M7-1 → M7-4 → M7-5 (issues #33, #36, #37).
4. **Do NOT** implement docker-compose.yml (#34) — blocked until M7-1 merges.

## Commits

Specialists do not commit. Propose granular commits per issue for human approval.

## Serial follow-up (after merges)

- deploy-engineer: #34 M7-2, #35 M7-3
- deploy-engineer: #38 M7-6 (blocker-reporter if domain undecided)
- docs-writer: #39 M7-7

Invoke **verifier** before each PR recommendation.
