#!/usr/bin/env bash
set -euo pipefail

# Run the full FPQC-SAC experiment matrix (main model, ablations, baselines, encoders).
# Set DATA and optional SEEDS/VIX_PATH before running. CUDA is used by default
# when available. Use DISABLE_VIX=1 only for no-VIX runs.
# Use DRY_RUN=1 to preview commands.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="${DATA:-}"
SEEDS="${SEEDS:-configs/seeds_repro_20.txt}"
VIX_PATH="${VIX_PATH:-data/raw/vix_panel.csv}"
DISABLE_VIX="${DISABLE_VIX:-0}"
PHASE="${PHASE:-train}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-20}"
THREADS_PER_PROC="${THREADS_PER_PROC:-4}"
RUN_PREFIX="${RUN_PREFIX:-fpqc_full_suite_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$DATA" ]]; then
  echo "DATA is required. Example: DATA=data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv bash scripts/run_full_suite.sh" >&2
  exit 2
fi

export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="$THREADS_PER_PROC"
export MKL_NUM_THREADS="$THREADS_PER_PROC"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROC"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROC"
export TORCH_NUM_THREADS="$THREADS_PER_PROC"
export PYTHONUNBUFFERED=1

cd "$ROOT"

SUITE_DIR="$ROOT/outputs/$RUN_PREFIX"
MANIFEST="$SUITE_DIR/manifest.csv"
mkdir -p "$SUITE_DIR"

COMMON_BASE=(
  --data-path "$DATA"
  --fix-seed-list-file "$SEEDS"
  --timesteps 25000
  --checkpoint-every 25000
  --train-start 2013-01-02
  --train-end 2018-12-31
  --test-start 2019-01-02
  --test-end 2021-08-31
  --simple-mcts-reward-delay-steps 10
  --observation-history-window 1
  --disable-lookahead
  --reward-scaling 0.0001
  --cash-scale-factor 1.0
  --learning-rate 0.0003
  --ent-coef 0.0
  --batch-size 64
  --buffer-size 200000
  --n-steps 1024
  --reward-log-every 1024
  --learn-log-interval 1
  --sac-mlp-bottleneck-dim 7
  --sac-weight-decay 0.0001
  --sac-n-qubits 7
  --sac-quantum-n-layers 2
  --sac-quantum-device cpu
  --sac-quantum-entanglement-topology ring
  --sac-quantum-embedding-type angle
  --sac-use-original-train
  --parallel-workers "$PARALLEL_WORKERS"
  --name-prefix baseline
  --model-index-start 0
)

if [[ "$DISABLE_VIX" == "1" ]]; then
  COMMON_BASE+=(--disable-vix)
elif [[ -n "$VIX_PATH" ]]; then
  COMMON_BASE+=(--vix-path "$VIX_PATH")
fi

EVAL_BASE=(
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
)

if [[ "$DISABLE_VIX" == "1" ]]; then
  EVAL_BASE+=(--disable-vix)
elif [[ -n "$VIX_PATH" ]]; then
  EVAL_BASE+=(--vix-path "$VIX_PATH")
fi

write_manifest() {
  cat > "$MANIFEST" <<EOF
model_key,label,category,train_kind,train_algo,eval_algo,tag,models_root,eval_dir,log_dir,order
fpqc_sac,FPQC-SAC,Proposed,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac,outputs/${RUN_PREFIX}/models/fpqc_sac,outputs/${RUN_PREFIX}/eval/fpqc_sac,outputs/${RUN_PREFIX}/logs/fpqc_sac,10
fpqc_no_cnot,No-CNOT FPQC-SAC,Core Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_no_cnot,outputs/${RUN_PREFIX}/models/fpqc_no_cnot,outputs/${RUN_PREFIX}/eval/fpqc_no_cnot,outputs/${RUN_PREFIX}/logs/fpqc_no_cnot,20
fpqc_frozen,Frozen-PQC FPQC-SAC,Core Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_frozen,outputs/${RUN_PREFIX}/models/fpqc_frozen,outputs/${RUN_PREFIX}/eval/fpqc_frozen,outputs/${RUN_PREFIX}/logs/fpqc_frozen,30
sac,SAC,DRL-based,baseline,sac,sac,${RUN_PREFIX}_sac,outputs/${RUN_PREFIX}/models/sac,outputs/${RUN_PREFIX}/eval/sac,outputs/${RUN_PREFIX}/logs/sac,40
td3,TD3,DRL-based,baseline,td3,td3,${RUN_PREFIX}_td3,outputs/${RUN_PREFIX}/models/td3,outputs/${RUN_PREFIX}/eval/td3,outputs/${RUN_PREFIX}/logs/td3,50
ddpg,DDPG,DRL-based,baseline,ddpg,ddpg,${RUN_PREFIX}_ddpg,outputs/${RUN_PREFIX}/models/ddpg,outputs/${RUN_PREFIX}/eval/ddpg,outputs/${RUN_PREFIX}/logs/ddpg,60
a2c,A2C,DRL-based,baseline,a2c,a2c,${RUN_PREFIX}_a2c,outputs/${RUN_PREFIX}/models/a2c,outputs/${RUN_PREFIX}/eval/a2c,outputs/${RUN_PREFIX}/logs/a2c,70
ppo,PPO,DRL-based,baseline,ppo,ppo,${RUN_PREFIX}_ppo,outputs/${RUN_PREFIX}/models/ppo,outputs/${RUN_PREFIX}/eval/ppo,outputs/${RUN_PREFIX}/logs/ppo,80
tqc,TQC,DRL-based,baseline,tqc,tqc,${RUN_PREFIX}_tqc,outputs/${RUN_PREFIX}/models/tqc,outputs/${RUN_PREFIX}/eval/tqc,outputs/${RUN_PREFIX}/logs/tqc,90
linear_bottleneck,Linear Bottleneck,Bottleneck,baseline,sac_linear_bottleneck,sac_linear_bottleneck,${RUN_PREFIX}_linear_bottleneck,outputs/${RUN_PREFIX}/models/linear_bottleneck,outputs/${RUN_PREFIX}/eval/linear_bottleneck,outputs/${RUN_PREFIX}/logs/linear_bottleneck,100
weight_decay_bottleneck,Weight Decay Bot.,Bottleneck,baseline,sac_weight_decay_bottleneck,sac_weight_decay_bottleneck,${RUN_PREFIX}_weight_decay_bottleneck,outputs/${RUN_PREFIX}/models/weight_decay_bottleneck,outputs/${RUN_PREFIX}/eval/weight_decay_bottleneck,outputs/${RUN_PREFIX}/logs/weight_decay_bottleneck,110
mlp_bottleneck,MLP Bottleneck,Bottleneck,baseline,sac_mlp,sac_mlp,${RUN_PREFIX}_mlp_bottleneck,outputs/${RUN_PREFIX}/models/mlp_bottleneck,outputs/${RUN_PREFIX}/eval/mlp_bottleneck,outputs/${RUN_PREFIX}/logs/mlp_bottleneck,120
spectral_bottleneck,SpectralNorm Bot.,Bottleneck,baseline,sac_spectral_bottleneck,sac_spectral_bottleneck,${RUN_PREFIX}_spectral_bottleneck,outputs/${RUN_PREFIX}/models/spectral_bottleneck,outputs/${RUN_PREFIX}/eval/spectral_bottleneck,outputs/${RUN_PREFIX}/logs/spectral_bottleneck,130
tanh_bottleneck,Tanh Bottleneck,Bottleneck,baseline,sac_tanh_bottleneck,sac_tanh_bottleneck,${RUN_PREFIX}_tanh_bottleneck,outputs/${RUN_PREFIX}/models/tanh_bottleneck,outputs/${RUN_PREFIX}/eval/tanh_bottleneck,outputs/${RUN_PREFIX}/logs/tanh_bottleneck,140
clipped_bottleneck,Clipped Latent,Bottleneck,baseline,sac_clipped_bottleneck,sac_clipped_bottleneck,${RUN_PREFIX}_clipped_bottleneck,outputs/${RUN_PREFIX}/models/clipped_bottleneck,outputs/${RUN_PREFIX}/eval/clipped_bottleneck,outputs/${RUN_PREFIX}/logs/clipped_bottleneck,150
layernorm_bottleneck,LayerNorm Bot.,Bottleneck,baseline,sac_layernorm_bottleneck,sac_layernorm_bottleneck,${RUN_PREFIX}_layernorm_bottleneck,outputs/${RUN_PREFIX}/models/layernorm_bottleneck,outputs/${RUN_PREFIX}/eval/layernorm_bottleneck,outputs/${RUN_PREFIX}/logs/layernorm_bottleneck,160
fpqc_sac_l1,FPQC-SAC-L1,Layer Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac_l1,outputs/${RUN_PREFIX}/models/fpqc_sac_l1,outputs/${RUN_PREFIX}/eval/fpqc_sac_l1,outputs/${RUN_PREFIX}/logs/fpqc_sac_l1,160
fpqc_sac_l2,FPQC-SAC-L2,Layer Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac_l2,outputs/${RUN_PREFIX}/models/fpqc_sac_l2,outputs/${RUN_PREFIX}/eval/fpqc_sac_l2,outputs/${RUN_PREFIX}/logs/fpqc_sac_l2,170
fpqc_sac_l3,FPQC-SAC-L3,Layer Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac_l3,outputs/${RUN_PREFIX}/models/fpqc_sac_l3,outputs/${RUN_PREFIX}/eval/fpqc_sac_l3,outputs/${RUN_PREFIX}/logs/fpqc_sac_l3,180
fpqc_sac_l4,FPQC-SAC-L4,Layer Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac_l4,outputs/${RUN_PREFIX}/models/fpqc_sac_l4,outputs/${RUN_PREFIX}/eval/fpqc_sac_l4,outputs/${RUN_PREFIX}/logs/fpqc_sac_l4,190
fpqc_sac_l5,FPQC-SAC-L5,Layer Ablation,baseline,sac,sac,${RUN_PREFIX}_fpqc_sac_l5,outputs/${RUN_PREFIX}/models/fpqc_sac_l5,outputs/${RUN_PREFIX}/eval/fpqc_sac_l5,outputs/${RUN_PREFIX}/logs/fpqc_sac_l5,200
wavelet,Wavelet,Encoder,baseline,sac_wavelet,sac_wavelet,${RUN_PREFIX}_wavelet,outputs/${RUN_PREFIX}/models/wavelet,outputs/${RUN_PREFIX}/eval/wavelet,outputs/${RUN_PREFIX}/logs/wavelet,210
kalman,Kalman,Encoder,baseline,sac_kalman,sac_kalman,${RUN_PREFIX}_kalman,outputs/${RUN_PREFIX}/models/kalman,outputs/${RUN_PREFIX}/eval/kalman,outputs/${RUN_PREFIX}/logs/kalman,260
fourier,RFF,Encoder,baseline,sac_fourier,sac_fourier,${RUN_PREFIX}_fourier,outputs/${RUN_PREFIX}/models/fourier,outputs/${RUN_PREFIX}/eval/fourier,outputs/${RUN_PREFIX}/logs/fourier,310
EOF
}

run_baseline_group() {
  local algo="$1"; shift
  local model_key="$1"; shift
  local tag="${RUN_PREFIX}_${model_key}"
  echo "[$(date -Iseconds)] train $tag ($algo)"
  local cmd=("$PYTHON_BIN" main_baselines_randomseed.py "${COMMON_BASE[@]}" \
    --algorithms "$algo" \
    --run-tag "$tag" \
    --model-dir "$ROOT/outputs/${RUN_PREFIX}/models/${model_key}/${algo}" \
    --tensorboard-log "$ROOT/outputs/${RUN_PREFIX}/logs/${model_key}/tb" \
    "$@")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}" 2>&1 | tee "$SUITE_DIR/train_${tag}.out"
  fi
}

train_all() {
  run_baseline_group sac fpqc_sac --sac-use-quantum-feature
  run_baseline_group sac fpqc_no_cnot --sac-use-quantum-feature --sac-quantum-no-entanglement
  run_baseline_group sac fpqc_frozen --sac-use-quantum-feature --sac-quantum-freeze-pqc

  for layer in 1 2 3 4 5; do
    run_baseline_group sac "fpqc_sac_l${layer}" --sac-use-quantum-feature --sac-quantum-n-layers "$layer"
  done

  run_baseline_group sac sac
  run_baseline_group td3 td3
  run_baseline_group ddpg ddpg
  run_baseline_group a2c a2c
  run_baseline_group ppo ppo
  run_baseline_group tqc tqc

  run_baseline_group sac_linear_bottleneck linear_bottleneck
  run_baseline_group sac_weight_decay_bottleneck weight_decay_bottleneck
  run_baseline_group sac_mlp mlp_bottleneck
  run_baseline_group sac_spectral_bottleneck spectral_bottleneck
  run_baseline_group sac_tanh_bottleneck tanh_bottleneck
  run_baseline_group sac_clipped_bottleneck clipped_bottleneck
  run_baseline_group sac_layernorm_bottleneck layernorm_bottleneck

  run_baseline_group sac_wavelet wavelet
  run_baseline_group sac_kalman kalman
  run_baseline_group sac_fourier fourier
}

eval_all() {
  while IFS=, read -r model_key label category train_kind train_algo eval_algo tag models_root eval_dir log_dir order; do
    [[ "$model_key" == "model_key" ]] && continue
    echo "[$(date -Iseconds)] eval $tag ($eval_algo)"
    local cmd=("$PYTHON_BIN" scripts/eval_baseline_models.py "${EVAL_BASE[@]}" \
      --models-root "$ROOT/$models_root" \
      --algorithms "$eval_algo" \
      --output-dir "$ROOT/$eval_dir" \
      --group-name "$model_key")
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}" 2>&1 | tee "$SUITE_DIR/eval_${tag}.out"
    fi
  done < "$MANIFEST"

  echo "[$(date -Iseconds)] rule-based backtests"
  local rule_cmd=("$PYTHON_BIN" main_baselines_randomseed.py "${COMMON_BASE[@]}" \
    --algorithms macd,kdj_rsi,zmr,sma \
    --rule-output-dir "$ROOT/outputs/${RUN_PREFIX}/eval/rules" \
    --model-dir "$ROOT/outputs/${RUN_PREFIX}/models/rules" \
    --tensorboard-log "$ROOT/outputs/${RUN_PREFIX}/logs/rules/tb")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${rule_cmd[@]}"
    printf '\n'
  else
    "${rule_cmd[@]}" 2>&1 | tee "$SUITE_DIR/eval_rules.out"
  fi
}

report_all() {
  local perf_cmd=("$PYTHON_BIN" scripts/build_trim2_metric_table.py \
    --manifest "$MANIFEST" \
    --output-dir "$SUITE_DIR/report/performance" \
    --caption "Trim-2 full-suite performance comparison." \
    --label "tab:trim2_full_suite_performance")
  local diag_cmd=("$PYTHON_BIN" scripts/build_training_diagnostic_table.py \
    --config configs/main_fpqc_sac.yaml \
    --config configs/baseline_sac.yaml \
    --config configs/ablations.yaml \
    --config configs/encoders_bottlenecks.yaml \
    --output-dir "$SUITE_DIR/report/diagnostics" \
    --caption "Trim-2 full-suite training diagnostics." \
    --label "tab:trim2_full_suite_diagnostics")
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${perf_cmd[@]}"
    printf '\n'
    printf '%q ' "${diag_cmd[@]}"
    printf '\n'
  else
    "${perf_cmd[@]}"
    if ! "${diag_cmd[@]}"; then
      echo "WARNING: diagnostic report was not generated. Performance tables are still available at $SUITE_DIR/report/performance." >&2
      echo "         This usually means the selected run did not produce compatible SAC diagnostic logs." >&2
    fi
  fi
}

write_manifest
echo "RUN_PREFIX=$RUN_PREFIX"
echo "SUITE_DIR=$SUITE_DIR"
echo "MANIFEST=$MANIFEST"
echo "PHASE=$PHASE PARALLEL_WORKERS=$PARALLEL_WORKERS THREADS_PER_PROC=$THREADS_PER_PROC"
echo "VIX_PATH=$VIX_PATH DISABLE_VIX=$DISABLE_VIX"

case "$PHASE" in
  train) train_all ;;
  eval) eval_all ;;
  report) report_all ;;
  all) train_all; eval_all; report_all ;;
  *) echo "Unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
