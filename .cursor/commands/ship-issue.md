# Ship one GitHub issue

Act as **milestone-orchestrator**. Ship the GitHub issue specified below (default: ask me for issue number if not provided).

## Steps

1. `gh issue view <NUMBER> --json title,body,labels,milestone`
2. Read `docs/ROADMAP.md` and `docs/PROJECT-DIRECTION.md` for that issue’s phase.
3. Read `AGENTS.md` issue→agent mapping for this number.
4. Create branch `feat/<milestone-short>-<name>` from latest `main`.
5. Dispatch specialists **only** for this issue (see AGENTS.md map).
6. Specialists must **NOT** run `git commit`.
7. Invoke **verifier** — `pytest tests/`; docker build only if Dockerfile/compose changed.
8. Split into **1–2 granular commits** per ROADMAP; show messages; commit **only if I said commit**.
9. Draft PR with **`## Main contribution`** first, then Summary, Test plan, `Closes #NN`. **Do not push** unless I say push.

### PR body template

```markdown
## Main contribution

<One paragraph: outcome and why it matters.>

## Summary
- ...

## Test plan
- [ ] ...

Closes #NN
```

## After merge (human)

- 15 min: read diff; run one manual test; note learning in `docs-private/`.
- Move issue to Done on GitHub Project.
- Next issue → **new Agent chat**.

## If blocked

Invoke **blocker-reporter**, use AGENTS.md Human decisions log, then **STOP**.

## Issue number

$ARGUMENTS
