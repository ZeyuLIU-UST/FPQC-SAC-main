#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/local_encoders_bottlenecks.yaml}"
STAGE="${STAGE:-train}"
EXPERIMENT="${EXPERIMENT:-all}"
DRY_RUN="${DRY_RUN:-0}"

args=(--config "$CONFIG" --experiment "$EXPERIMENT" --stage "$STAGE")
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" run.py "${args[@]}"
