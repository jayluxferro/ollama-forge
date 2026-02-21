#!/usr/bin/env sh
# Fix lint: ruff check --fix, then ruff format. Run before push to clear all fixable issues.
# Usage: ./scripts/lint-fix.sh   or from repo root: uv run ruff check src tests --fix && uv run ruff format src tests

set -e
cd "$(dirname "$0")/.."

echo "Running ruff check --fix..."
uv run ruff check src tests --fix
echo "Running ruff format..."
uv run ruff format src tests
echo "Verifying ruff check..."
if ! uv run ruff check src tests; then
  echo "Ruff still reports issues (e.g. line length). Fix them and run again."
  exit 1
fi
echo "Lint-fix done. You can commit/push."
