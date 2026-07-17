#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/upstream"

SECONDS=0
../.venv/bin/python -u anpm/experiments/anpm_amazon.py \
  --T 100 \
  --k 30 \
  --sigma 1e-3 \
  --exp_name sigma1e-3_k30_every10 \
  --error_every 10
echo "WALL_SECONDS=$SECONDS"
