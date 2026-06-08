#!/usr/bin/env bash
set -euo pipefail

# Export encoder features, build latent-variance/VAF inputs, and plot VAF@1.
# This script assumes the extended suite has already trained the listed models.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="${DATA:-data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv}"
RUN_SUFFIX="${RUN_SUFFIX:-25k_rewardcache_cuda_extended}"
RUN_PREFIX="${RUN_PREFIX:-repro_mainstream_tech_market_index_extended_${RUN_SUFFIX}}"
VIX_PATH="${VIX_PATH:-data/raw/vix_panel.csv}"
DISABLE_VIX="${DISABLE_VIX:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING_FEATURES="${SKIP_EXISTING_FEATURES:-1}"
FIGURE_GROUP_SET="${FIGURE_GROUP_SET:-all}"  # encoder | bottleneck | all

MANIFEST="${MANIFEST:-outputs/${RUN_PREFIX}/manifest.csv}"
FEATURE_EVAL_ROOT="${FEATURE_EVAL_ROOT:-outputs/${RUN_PREFIX}/eval_features}"
FIGURE_INPUT_DIR="${FIGURE_INPUT_DIR:-outputs/figures/input}"

if [[ -z "${VAF_GROUPS+x}" ]]; then
  case "$FIGURE_GROUP_SET" in
    encoder)
      VAF_GROUPS="fpqc_sac,wavelet,kalman,fourier"
      ;;
    bottleneck)
      VAF_GROUPS="fpqc_sac,sac,linear_bottleneck,weight_decay_bottleneck,mlp_bottleneck,spectral_bottleneck,tanh_bottleneck,clipped_bottleneck,layernorm_bottleneck"
      ;;
    all)
      VAF_GROUPS="fpqc_sac,sac,linear_bottleneck,weight_decay_bottleneck,mlp_bottleneck,spectral_bottleneck,tanh_bottleneck,clipped_bottleneck,layernorm_bottleneck,wavelet,kalman,fourier"
      ;;
    *)
      echo "Unknown FIGURE_GROUP_SET=$FIGURE_GROUP_SET. Use encoder, bottleneck, or all." >&2
      exit 2
      ;;
  esac
fi

COMBINED_FEATURE_CSV="${COMBINED_FEATURE_CSV:-${FIGURE_INPUT_DIR}/baseline_all_model_feature_vectors_${FIGURE_GROUP_SET}.csv}"
VARIANCE_CSV="${VARIANCE_CSV:-${FIGURE_INPUT_DIR}/variance_by_group_${FIGURE_GROUP_SET}.csv}"
VAF_CSV="${VAF_CSV:-${FIGURE_INPUT_DIR}/overall_feature_shape_vaf_by_group_${FIGURE_GROUP_SET}.csv}"
OUT_PNG="${OUT_PNG:-outputs/figures/latent_variance_vaf1_${FIGURE_GROUP_SET}.png}"

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$ROOT" "$path"
  fi
}

MANIFEST_ABS="$(resolve_path "$MANIFEST")"
FEATURE_EVAL_ROOT_ABS="$(resolve_path "$FEATURE_EVAL_ROOT")"
COMBINED_FEATURE_CSV_ABS="$(resolve_path "$COMBINED_FEATURE_CSV")"
VARIANCE_CSV_ABS="$(resolve_path "$VARIANCE_CSV")"
VAF_CSV_ABS="$(resolve_path "$VAF_CSV")"
OUT_PNG_ABS="$(resolve_path "$OUT_PNG")"

echo "ROOT=$ROOT"
echo "DATA=$DATA"
echo "RUN_PREFIX=$RUN_PREFIX"
echo "MANIFEST=$MANIFEST_ABS"
echo "FIGURE_GROUP_SET=$FIGURE_GROUP_SET"
echo "VAF_GROUPS=$VAF_GROUPS"
echo "SKIP_EXISTING_FEATURES=$SKIP_EXISTING_FEATURES"

if [[ "$DRY_RUN" != "1" ]]; then
  for f in "$DATA" "$MANIFEST_ABS"; do
    if [[ ! -f "$f" ]]; then
      echo "Missing required file: $f" >&2
      echo "Train/evaluate the extended suite first, or set RUN_PREFIX and DATA to existing outputs." >&2
      exit 1
    fi
  done
  if [[ "$DISABLE_VIX" != "1" && ! -f "$VIX_PATH" ]]; then
    echo "Missing VIX file: $VIX_PATH" >&2
    echo "Set DISABLE_VIX=1 only for no-VIX checks." >&2
    exit 1
  fi
fi

IFS=',' read -r -a REQUESTED_GROUPS <<< "$VAF_GROUPS"

contains_group() {
  local needle="$1"
  local group
  for group in "${REQUESTED_GROUPS[@]}"; do
    [[ "$group" == "$needle" ]] && return 0
  done
  return 1
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

mkdir -p "$FEATURE_EVAL_ROOT_ABS" "$(dirname "$COMBINED_FEATURE_CSV_ABS")" "$(dirname "$OUT_PNG_ABS")"

feature_csvs=()
while IFS=, read -r model_key label category train_kind train_algo eval_algo tag models_root eval_dir log_dir order; do
  [[ "$model_key" == "model_key" ]] && continue
  if ! contains_group "$model_key"; then
    continue
  fi

  models_path="$ROOT/$models_root"
  out_dir="${FEATURE_EVAL_ROOT_ABS}/${model_key}"
  feature_csv="$out_dir/baseline_all_model_feature_vectors.csv"
  feature_csvs+=("$feature_csv")

  if [[ "$SKIP_EXISTING_FEATURES" == "1" && -s "$feature_csv" ]]; then
    echo "[$(date -Iseconds)] reuse feature eval $model_key: $feature_csv"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" scripts/eval_baseline_models.py
    --models-root "$models_path"
    --algorithms "$eval_algo"
    --data-path "$DATA"
    --train-start 2013-01-02
    --train-end 2018-12-31
    --test-start 2019-01-02
    --test-end 2021-08-31
    --simple-mcts-reward-delay-steps 10
    --observation-history-window 1
    --disable-lookahead
    --reward-scaling 0.0001
    --cash-scale-factor 1.0
    --output-dir "$out_dir"
    --group-name "$model_key"
    --export-feature-vectors
  )
  if [[ "$DISABLE_VIX" == "1" ]]; then
    cmd+=(--disable-vix)
  else
    cmd+=(--vix-path "$VIX_PATH")
  fi

  echo "[$(date -Iseconds)] feature eval $model_key"
  run_cmd "${cmd[@]}"
done < "$MANIFEST_ABS"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: skipped feature merge and plotting."
  exit 0
fi

"$PYTHON_BIN" - "$COMBINED_FEATURE_CSV_ABS" "${feature_csvs[@]}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

out = Path(sys.argv[1])
inputs = [Path(path) for path in sys.argv[2:]]
frames = []
missing = []
for path in inputs:
    if path.is_file():
        frames.append(pd.read_csv(path))
    else:
        missing.append(str(path))
if missing:
    raise SystemExit("Missing feature CSV files:\n" + "\n".join(missing))
if not frames:
    raise SystemExit("No feature CSV files were produced.")
out.parent.mkdir(parents=True, exist_ok=True)
pd.concat(frames, ignore_index=True).to_csv(out, index=False)
print(out)
PY

"$PYTHON_BIN" scripts/build_latent_variance_vaf1_inputs.py \
  --feature-csv "$COMBINED_FEATURE_CSV_ABS" \
  --variance-csv "$VARIANCE_CSV_ABS" \
  --vaf-csv "$VAF_CSV_ABS" \
  --groups "$VAF_GROUPS"

"$PYTHON_BIN" scripts/plot_latent_variance_vaf1.py \
  --variance-csv "$VARIANCE_CSV_ABS" \
  --vaf-csv "$VAF_CSV_ABS" \
  --out "$OUT_PNG_ABS"

echo "combined_features=$COMBINED_FEATURE_CSV_ABS"
echo "variance_csv=$VARIANCE_CSV_ABS"
echo "vaf_csv=$VAF_CSV_ABS"
echo "figure=$OUT_PNG_ABS"
