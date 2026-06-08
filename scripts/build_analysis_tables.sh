#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-outputs/analysis_tables}"
CONFIGS=(${CONFIGS:-configs/main_fpqc_sac.yaml configs/baseline_sac.yaml configs/encoders_bottlenecks.yaml configs/ablations.yaml})
OOS_CONFIG="${OOS_CONFIG:-configs/oos_baseline_curves.yaml}"
EXTENDED_MANIFEST_GLOB="${EXTENDED_MANIFEST_GLOB:-outputs/repro_*_extended_*/manifest.csv}"
MANIFESTS="${MANIFESTS:-}"

manifest_list=()
if [[ -n "$MANIFESTS" ]]; then
  read -r -a manifest_list <<< "$MANIFESTS"
else
  while IFS= read -r manifest; do
    manifest_list+=("$manifest")
  done < <(compgen -G "$EXTENDED_MANIFEST_GLOB" | sort)
fi

if [[ "${#manifest_list[@]}" -gt 0 ]]; then
  echo "Using extended-suite manifest(s) for performance tables."
  for manifest in "${manifest_list[@]}"; do
    if [[ ! -f "$manifest" ]]; then
      echo "Missing manifest: $manifest" >&2
      exit 1
    fi
    run_prefix="$(basename "$(dirname "$manifest")")"
    "$PYTHON_BIN" scripts/build_trim2_metric_table.py \
      --manifest "$manifest" \
      --output-dir "$OUT_DIR/$run_prefix/performance" \
      --caption "Trim-2 full-suite performance comparison for $run_prefix." \
      --label "tab:trim2_${run_prefix}_performance"
  done
  echo "Done. Tables are under $OUT_DIR/<run_prefix>/performance/."
  echo "Set MANIFESTS='path/to/manifest.csv ...' to select specific runs."
  exit 0
fi

echo "No extended-suite manifests found; falling back to config-based outputs."

"$PYTHON_BIN" scripts/build_trim2_metric_table.py \
  --output-dir "$OUT_DIR/in_sample_performance" \
  --caption "Trim-2 in-sample test performance comparison." \
  --label "tab:trim2_in_sample_performance" \
  $(printf ' --config %q' "${CONFIGS[@]}")

"$PYTHON_BIN" scripts/build_trim2_metric_table.py \
  --config "$OOS_CONFIG" \
  --eval-subdir eval_oos \
  --output-dir "$OUT_DIR/oos_baseline_curves" \
  --caption "Trim-2 OOS baseline performance comparison." \
  --label "tab:trim2_oos_baseline_curves"

"$PYTHON_BIN" scripts/build_training_diagnostic_table.py \
  --output-dir "$OUT_DIR/training_diagnostics" \
  --caption "Trim-2 training diagnostics for SAC-family models." \
  --label "tab:trim2_training_diagnostics" \
  $(printf ' --config %q' "${CONFIGS[@]}")
