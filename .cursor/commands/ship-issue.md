# Ship one GitHub issue

Act as **milestone-orchestrator**. Ship the GitHub issue specified below (default: ask me for issue number if not provided).

Follow **train mode** in `AGENTS.md` unless the operator said `hold merges` / `propose only`.

## Steps

1. `gh issue view <NUMBER> --json title,body,labels,milestone`
2. Read `docs/operators/ROADMAP.md` and `docs/operators/PROJECT-DIRECTION.md` for that issue’s phase.
3. Read `AGENTS.md` issue→agent mapping for this number.
4. Create branch `feat/<milestone-short>-<name>` from latest `main`.
5. Dispatch specialists **only** for this issue (see AGENTS.md map).
6. Specialists must **NOT** run `git commit`.
7. Invoke **verifier**: `pytest tests/`; docker build only if Dockerfile/compose changed.
8. Split into **1–2 granular commits** per ROADMAP; commit after verifier green (train mode).
9. Open PR with **`## Main contribution`** first, then Summary, Test plan, `Closes #NN`.
10. **Train mode:** push, wait for CI green + checklist, merge, emit a **status pulse**, stop only on hard gates.
11. **Hold-merges mode:** draft PR; wait for explicit `commit` / `push` / `merge`.

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

### Status pulse template

```markdown
## Pulse · #NN merged
- Done: <one outcome line>
- PR: #<pr> → closes #NN
- Next: <next issue or parallel pair>
- Need from you: nothing | see Blocker
```

## After merge

- Emit status pulse (required).
- Optional human learning pass later; do **not** gate the next train issue on it.
- Move issue to Done on GitHub Project when possible.
- If continuing a train, proceed to the next wave in the same chat unless context is too large.

## If blocked

Invoke **blocker-reporter**, ask **docs-writer** to polish the Blocker card, use AGENTS.md Human decisions log, then **STOP**.

Hard gates: #57 video URL, secrets, CI still red after one fix attempt.

## Issue number

$ARGUMENTS
