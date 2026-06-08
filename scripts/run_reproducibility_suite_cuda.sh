#!/usr/bin/env bash
set -euo pipefail

# Fixed reproduction suite. Runs the three data groups in order:
#   FPQC-SAC, PPO, DDPG, SAC, TD3, A2C, TQC
# CUDA is required. By default, the script accepts NVIDIA RTX 4090 or L40 GPUs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-configs/seeds_repro_20.txt}"
VIX_PATH="${VIX_PATH:-data/raw/vix_panel.csv}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-5}"
THREADS_PER_PROC="${THREADS_PER_PROC:-1}"
PHASE="${PHASE:-all}"  # train | eval | all
RUN_SUFFIX="${RUN_SUFFIX:-25k_rewardcache_cuda_repro}"
ONLY_GROUP="${ONLY_GROUP:-}"  # mainstream_tech_market_index|defensive_blue_chip|high_volatility_growth
ONLY_ALGO="${ONLY_ALGO:-}"    # fpqc-sac|ppo|ddpg|sac|td3|a2c|tqc
DRY_RUN="${DRY_RUN:-0}"
ALLOW_OTHER_GPU="${ALLOW_OTHER_GPU:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS_PER_PROC"
export MKL_NUM_THREADS="$THREADS_PER_PROC"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROC"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROC"
export TORCH_NUM_THREADS="$THREADS_PER_PROC"

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
import sys
import torch

allow_other = os.environ.get("ALLOW_OTHER_GPU", "0") == "1"
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if torch.version.cuda is None:
    raise SystemExit("PyTorch is not a CUDA build. Install the CUDA environment before running.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to run the reproduction suite.")
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

COMMON_ARGS=(
  --fix-seed-list-file "$SEEDS"
  --parallel-workers "$PARALLEL_WORKERS"
  --timesteps 25000
  --checkpoint-every 25000
  --train-start 2013-01-02
  --train-end 2018-12-31
  --test-start 2019-01-02
  --test-end 2021-08-31
  --lookahead-window 5
  --simple-mcts-reward-delay-steps 10
  --disable-lookahead
  --reward-scaling 0.0001
  --learning-rate 0.0003
  --batch-size 64
  --buffer-size 200000
  --reward-log-every 1024
  --learn-log-interval 4
  --vix-path "$VIX_PATH"
)

FPQC_SAC_ARGS=(
  --sac-use-quantum-feature
  --sac-n-qubits 7
  --sac-quantum-n-layers 2
  --sac-quantum-device cpu
  --sac-quantum-entanglement-topology ring
  --sac-quantum-embedding-type angle
)

EVAL_ARGS=(
  --train-start 2013-01-02
  --train-end 2018-12-31
  --test-start 2019-01-02
  --test-end 2021-08-31
  --lookahead-window 5
  --simple-mcts-reward-delay-steps 10
  --disable-lookahead
  --reward-scaling 0.0001
  --vix-path "$VIX_PATH"
)

algo_allowed() {
  local algo="$1"
  [[ -z "$ONLY_ALGO" || "$ONLY_ALGO" == "$algo" ]]
}

group_allowed() {
  local group="$1"
  [[ -z "$ONLY_GROUP" || "$ONLY_GROUP" == "$group" ]]
}

set_algo_extra_args() {
  local algo="$1"
  ALGO_EXTRA_ARGS=()
  case "$algo" in
    ppo)
      ALGO_EXTRA_ARGS=(--n-steps 2048 --ent-coef 0.01)
      ;;
    fpqc-sac|sac|ddpg|td3|a2c|tqc)
      ALGO_EXTRA_ARGS=(--n-steps 1024 --ent-coef 0.0)
      ;;
    *)
      echo "Unsupported algo: $algo" >&2
      return 1
      ;;
  esac
}

tag_for() {
  local group="$1"
  local algo="$2"
  case "$algo" in
    fpqc-sac) echo "repro_${group}_fpqc_sac_q7_${RUN_SUFFIX}" ;;
    *) echo "repro_${group}_${algo}_classic_${RUN_SUFFIX}" ;;
  esac
}

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_train_one() {
  local group="$1"
  local data="$2"
  local algo="$3"

  if ! algo_allowed "$algo"; then
    echo "=== skip train group=${group} algo=${algo} ==="
    return 0
  fi

  local tag
  tag="$(tag_for "$group" "$algo")"
  local model_root="$ROOT/outputs/${tag}/models/${algo}"
  local log_root="$ROOT/outputs/${tag}/logs"
  local train_algo="$algo"
  local -a quantum_args=()

  if [[ "$algo" == "fpqc-sac" ]]; then
    train_algo="sac"
    model_root="$ROOT/outputs/${tag}/models/sac"
    quantum_args=("${FPQC_SAC_ARGS[@]}")
  fi

  mkdir -p "$model_root" "$log_root/tb"
  set_algo_extra_args "$algo"

  echo "=== train group=${group} algo=${algo} tag=${tag} ==="
  local -a cmd=(
    "$PYTHON_BIN" main_baselines_randomseed.py
    --data-path "$data" \
    --algorithms "$train_algo" \
    "${COMMON_ARGS[@]}" \
    "${ALGO_EXTRA_ARGS[@]}" \
    --run-tag "$tag" \
    --tensorboard-log "$log_root/tb" \
    --model-dir "$model_root" \
    --name-prefix baseline
  )
  if [[ "${#quantum_args[@]}" -gt 0 ]]; then
    cmd+=("${quantum_args[@]}")
  fi
  run_or_print "${cmd[@]}"
}

run_eval_one() {
  local group="$1"
  local data="$2"
  local algo="$3"

  if ! algo_allowed "$algo"; then
    echo "=== skip eval group=${group} algo=${algo} ==="
    return 0
  fi

  local tag
  tag="$(tag_for "$group" "$algo")"
  local eval_algo="$algo"
  local models_root="$ROOT/outputs/${tag}/models"

  if [[ "$algo" == "fpqc-sac" ]]; then
    eval_algo="sac"
  else
    models_root="$ROOT/outputs/${tag}/models"
  fi

  echo "=== eval group=${group} algo=${algo} tag=${tag} ==="
  local -a cmd=(
    "$PYTHON_BIN" scripts/eval_baseline_models.py
    --data-path "$data" \
    "${EVAL_ARGS[@]}" \
    --models-root "$models_root" \
    --algorithms "$eval_algo" \
    --group-name "$algo" \
    --output-dir "$ROOT/outputs/${tag}/eval"
  )
  run_or_print "${cmd[@]}"
}

train_group() {
  local group="$1"
  local data="$2"
  run_train_one "$group" "$data" fpqc-sac
  run_train_one "$group" "$data" ppo
  run_train_one "$group" "$data" ddpg
  run_train_one "$group" "$data" sac
  run_train_one "$group" "$data" td3
  run_train_one "$group" "$data" a2c
  run_train_one "$group" "$data" tqc
}

eval_group() {
  local group="$1"
  local data="$2"
  run_eval_one "$group" "$data" fpqc-sac
  run_eval_one "$group" "$data" ppo
  run_eval_one "$group" "$data" ddpg
  run_eval_one "$group" "$data" sac
  run_eval_one "$group" "$data" td3
  run_eval_one "$group" "$data" a2c
  run_eval_one "$group" "$data" tqc
}

run_group() {
  local group="$1"
  local data="$2"

  if ! group_allowed "$group"; then
    echo "=== skip group=${group} ==="
    return 0
  fi

  echo "############################################################"
  echo "START group=${group} data=${data}"
  echo "############################################################"

  case "$PHASE" in
    train) train_group "$group" "$data" ;;
    eval) eval_group "$group" "$data" ;;
    all) train_group "$group" "$data"; eval_group "$group" "$data" ;;
    *) echo "Unsupported PHASE=$PHASE (train|eval|all)" >&2; exit 2 ;;
  esac

  echo "DONE group=${group}"
}

run_group mainstream_tech_market_index data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
run_group defensive_blue_chip data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
run_group high_volatility_growth data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv

echo "=== fixed CUDA reproduction suite completed ==="
