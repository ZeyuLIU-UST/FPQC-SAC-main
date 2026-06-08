#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/oos_baseline_curves.yaml}"
STAGE="${STAGE:-eval}"
EXPERIMENT="${EXPERIMENT:-all}"
DRY_RUN="${DRY_RUN:-0}"
DATA="${DATA:-}"

RUN_CONFIG="$CONFIG"
TMP_CONFIG=""
if [[ "$STAGE" == "train" || "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  if [[ -z "$DATA" ]]; then
    echo "DATA is required for STAGE=$STAGE. Example:" >&2
    echo "DATA=data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv STAGE=eval bash scripts/run_oos_baseline_curves.sh" >&2
    exit 2
  fi
  TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/fpqc_oos_config.XXXXXX")"
  "$PYTHON_BIN" - "$CONFIG" "$TMP_CONFIG" "$DATA" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data_path = sys.argv[3]
text = src.read_text(encoding="utf-8")
quoted = data_path.replace('"', '\\"')
updated = text.replace('data_path: ""', f'data_path: "{quoted}"', 2)
if updated == text:
    raise SystemExit("Could not inject DATA into config; expected two data_path: \"\" entries.")
dst.write_text(updated, encoding="utf-8")
PY
  RUN_CONFIG="$TMP_CONFIG"
fi

cleanup() {
  if [[ -n "$TMP_CONFIG" && -f "$TMP_CONFIG" ]]; then
    rm -f "$TMP_CONFIG"
  fi
}
trap cleanup EXIT

args=(--config "$RUN_CONFIG" --experiment "$EXPERIMENT" --stage "$STAGE")
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" run.py "${args[@]}"
