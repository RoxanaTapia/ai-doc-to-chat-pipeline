# Verify: tests and quality gate

Act as **verifier**. Run quality checks before a PR.

## Run

1. `pytest tests/ -v` from repo root
2. If `ruff` is available: `ruff check src tests` and `ruff format --check src tests`
3. If `Dockerfile` or `docker-compose*.yml` changed in this branch: `docker build -t ai-doc-to-chat-test .`
4. If `src/api/` exists: note whether `/health` should be curl-tested

## Output

Use the verifier report format from `.cursor/agents/verifier.md`:

- pytest PASS/FAIL
- ruff PASS/FAIL/SKIP
- docker build PASS/FAIL/N/A
- Test gaps
- Required fixes before PR

**Do not** implement feature fixes in `src/`, report which specialist should fix failures.

Optional scope: $ARGUMENTS
