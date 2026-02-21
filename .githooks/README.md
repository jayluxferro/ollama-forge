# Git hooks

These hooks run automatically when you use the repo’s Git config.

## Install (one-time)

From the repo root:

```bash
git config core.hooksPath .githooks
```

To install for all repos globally: `git config --global core.hooksPath ~/.githooks` (and copy this folder there if desired).

## Hooks

- **pre-commit** — Runs `ruff check src tests --fix` and `ruff format src tests` before each commit. If Ruff still reports issues after fixing, the commit is aborted. Fixed files are staged so the commit includes the fixes.
- **pre-push** — Runs `ruff check src tests` and `pytest tests/` before each push. If either fails, push is aborted. Fix lint first with `./scripts/lint-fix.sh`.
