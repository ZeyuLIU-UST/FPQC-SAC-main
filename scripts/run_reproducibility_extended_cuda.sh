#!/usr/bin/env bash
set -euo pipefail

# Extended fixed reproduction suite. This runs the full experiment matrix
# used for ablations, bottlenecks, encoders, DRL baselines, and reports.
# It uses the fixed three data groups and does not require YAML edits.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-configs/seeds_repro_20.txt}"
VIX_PATH="${VIX_PATH:-data/raw/vix_panel.csv}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-5}"
THREADS_PER_PROC="${THREADS_PER_PROC:-1}"
PHASE="${PHASE:-all}"  # train | eval | report | all
RUN_SUFFIX="${RUN_SUFFIX:-25k_rewardcache_cuda_extended}"
ONLY_GROUP="${ONLY_GROUP:-}"  # mainstream_tech_market_index|defensive_blue_chip|high_volatility_growth
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OTHER_GPU="${ALLOW_OTHER_GPU:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

cd "$ROOT"

required_files=(
  "$SEEDS"
  "$VIX_PATH"
  "data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv"
  "data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv"
  "data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv"
)

if [[ "$DRY_RUN" != "1" ]]; then
  for f in "${required_files[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "Missing required file: $f" >&2
      echo "Run: bash scripts/download_reproducibility_data.sh" >&2
      exit 1
    fi
  done
fi

echo "ROOT=$ROOT"
echo "SEEDS=$SEEDS"
echo "VIX_PATH=$VIX_PATH"
echo "PHASE=$PHASE"
echo "RUN_SUFFIX=$RUN_SUFFIX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PARALLEL_WORKERS=$PARALLEL_WORKERS"
echo "THREADS_PER_PROC=$THREADS_PER_PROC"
echo "ALLOW_OTHER_GPU=$ALLOW_OTHER_GPU"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: skipping CUDA and input-file availability checks."
else
  "$PYTHON_BIN" - <<'PY'
import os
import torch

allow_other = os.environ.get("ALLOW_OTHER_GPU", "0") == "1"
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if torch.version.cuda is None:
    raise SystemExit("PyTorch is not a CUDA build. Install the CUDA environment before running.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to run the extended reproduction suite.")
if torch.cuda.device_count() < 1:
    raise SystemExit("No visible CUDA device. Set CUDA_VISIBLE_DEVICES to an NVIDIA GPU.")

names = []
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    names.append(name)
    print("device", i, name)

allowed = any(("4090" in name) or ("L40" in name.upper()) for name in names)
if not allowed and not allow_other:
    raise SystemExit(
        "The reported runs use NVIDIA RTX 4090 or NVIDIA L40. "
        "Set ALLOW_OTHER_GPU=1 only for exploratory runs on other NVIDIA GPUs."
    )
PY
fi

group_allowed() {
  local group="$1"
  [[ -z "$ONLY_GROUP" || "$ONLY_GROUP" == "$group" ]]
}

run_group() {
  local group="$1"
  local data="$2"
  local prefix="repro_${group}_extended_${RUN_SUFFIX}"

  if ! group_allowed "$group"; then
    echo "=== skip group=${group} ==="
    return 0
  fi

  echo "############################################################"
  echo "START extended group=${group} data=${data}"
  echo "RUN_PREFIX=${prefix}"
  echo "############################################################"

  PYTHON_BIN="$PYTHON_BIN" \
  DATA="$data" \
  SEEDS="$SEEDS" \
  VIX_PATH="$VIX_PATH" \
  PARALLEL_WORKERS="$PARALLEL_WORKERS" \
  THREADS_PER_PROC="$THREADS_PER_PROC" \
  PHASE="$PHASE" \
  RUN_PREFIX="$prefix" \
  DRY_RUN="$DRY_RUN" \
  bash scripts/run_full_suite.sh

  echo "DONE extended group=${group}"
}

run_group mainstream_tech_market_index data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
run_group defensive_blue_chip data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
run_group high_volatility_growth data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv

echo "=== fixed CUDA extended reproduction suite completed ==="
