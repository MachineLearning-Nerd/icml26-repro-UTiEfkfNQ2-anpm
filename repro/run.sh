#!/usr/bin/env bash
# Fixed run command for the ANPM (UTiEfkfNQ2) reproduction.
# Bootstraps uv, syncs the pinned environment from uv.lock, and runs the master
# verifier. Identical on every experiment node (children inherit it verbatim).
set -euo pipefail
cd "$(dirname "$0")/.."

# Bootstrap uv if not present, then sync the locked environment.
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --quiet uv
fi
uv sync --frozen --quiet

# Run the full claim-by-claim verifier suite; prints all metrics to stdout.
uv run python repro/src/run_all.py
