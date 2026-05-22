# Ship one M7 GitHub issue

Act as **milestone-orchestrator**. Ship the M7 GitHub issue specified below (default: ask me for issue number if not provided).

## Steps

1. `gh issue view <NUMBER> --json title,body,labels,milestone`
2. Read `docs/ROADMAP.md` M7 section and `AGENTS.md` issue→agent mapping.
3. Create branch `feat/m7-<short-name>` from latest main (or current base if I specify).
4. Dispatch specialists **only** for this issue:

   | Issue | Primary | Secondary |
   |-------|---------|-----------|
   | #33 | deploy-engineer | — |
   | #34 | deploy-engineer | config-guardian |
   | #35 | deploy-engineer | docs-writer |
   | #36 | config-guardian | docs-writer |
   | #37 | docs-writer | deploy-engineer review |
   | #38 | deploy-engineer | blocker-reporter if no domain |
   | #39 | docs-writer | — |

5. Specialists must **NOT** run `git commit`.
6. Invoke **verifier** — pytest; docker build if Dockerfile/compose changed.
7. Split into **1–2 granular commits** per ROADMAP; show messages; commit **only if I said commit**.
8. Draft PR title/body with `Closes #NN`. **Do not push** unless I say push.

## If blocked

Invoke **blocker-reporter**, use AGENTS.md Human decisions log defaults, then **STOP**.

## Issue number

$ARGUMENTS
