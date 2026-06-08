#!/usr/bin/env python3
"""
Evaluate A2C/DDPG/TD3/SAC/TQC/PPO model directories and write summary CSV files.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Allow direct execution from scripts/ while importing project modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress repeated SB3 warnings when loading older models with list net_arch.
warnings.filterwarnings("ignore", message=".*net_arch.*SB3 v1.8.0.*", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
try:
    from sb3_contrib import TQC  # type: ignore
except Exception:
    TQC = None
from finrl.agents.stablebaselines3.models import DRLAgent
import fpqc_sac.repro_utils as mqr
import models  # noqa: F401


ALGO_CLASS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "sac_quantum": SAC,
    "sac_mlp": SAC,
    "sac_linear_bottleneck": SAC,
    "sac_tanh_bottleneck": SAC,
    "sac_clipped_bottleneck": SAC,
    "sac_layernorm_bottleneck": SAC,
    "sac_spectral_bottleneck": SAC,
    "sac_weight_decay_bottleneck": SAC,
    "sac_fourier": SAC,
    "sac_wavelet": SAC,
    "sac_kalman": SAC,
    "tqc": TQC,
    "ppo": PPO,
}


def _compute_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return float(annual_return) / (abs(float(max_drawdown)) + 1e-8)


def _extract_seed(name: str) -> int:
    m = re.search(r"_s(\d+)_", name)
    if not m:
        return 42
    return int(m.group(1))


def _extract_m_index(name: str) -> int:
    m = re.search(r"_m(\d+)", name)
    if not m:
        return 0
    return int(m.group(1))


def _resolve_group_name(models_root: Path, algo: str, cli_group_name: str | None) -> str:
    if cli_group_name:
        return cli_group_name
    algo_dir = models_root / algo
    if algo_dir.is_dir():
        return algo
    return models_root.name


def _evaluate_once(model, test_df: pd.DataFrame, env_kwargs: Dict, deterministic: bool) -> pd.DataFrame:
    eval_env = mqr.make_env(test_df, env_kwargs)
    account_memory, _ = DRLAgent.DRL_prediction(model, environment=eval_env, deterministic=deterministic)
    if isinstance(account_memory, pd.DataFrame):
        df = account_memory.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    out = pd.DataFrame({"date": eval_env.date_memory, "account_value": account_memory})
    out["date"] = pd.to_datetime(out["date"])
    return out


def _extract_model_features(model, obs: np.ndarray) -> np.ndarray:
    """
    Extract encoder features for the current observation.
    SAC-like models read actor.features_extractor first; PPO/A2C fall back to
    policy.extract_features.
    """
    with torch.no_grad():
        policy = model.policy
        policy.set_training_mode(False)
        obs_tensor, _ = policy.obs_to_tensor(obs)
        if hasattr(model, "actor") and hasattr(model.actor, "features_extractor"):
            features = model.actor.features_extractor(obs_tensor)
        else:
            features = policy.extract_features(obs_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features_np = features.detach().cpu().numpy()
    if features_np.ndim == 1:
        features_np = features_np.reshape(1, -1)
    return features_np


def _evaluate_once_with_features(
    model,
    test_df: pd.DataFrame,
    env_kwargs: Dict,
    deterministic: bool,
    group_name: str,
    algorithm: str,
    model_name: str,
    train_seed: int,
    eval_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one model and export encoder features for each test step.
    """
    eval_env = mqr.make_env(test_df, env_kwargs)
    test_env, test_obs = eval_env.get_sb_env()
    reset_obs = test_env.reset()
    if reset_obs is not None:
        test_obs = reset_obs

    account_memory = None
    max_steps = len(eval_env.df.index.unique()) - 1
    unique_dates = pd.to_datetime(eval_env.df["date"].drop_duplicates().tolist())
    feature_rows: List[Dict] = []

    for i in range(len(eval_env.df.index.unique())):
        features_np = _extract_model_features(model, test_obs)
        if len(features_np) > 0:
            vec = features_np[0]
            row: Dict = {
                "group_name": group_name,
                "algorithm": algorithm,
                "model_name": model_name,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "step": i,
                "date": pd.Timestamp(unique_dates[i]).strftime("%Y-%m-%d") if i < len(unique_dates) else None,
            }
            for j, value in enumerate(vec):
                row[f"f{j}"] = float(value)
            feature_rows.append(row)

        action, _states = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)

        if i == max_steps - 1:
            account_memory = test_env.env_method(method_name="save_asset_memory")

        if dones[0]:
            print("hit end!")
            break

    if account_memory is None:
        account_memory = test_env.env_method(method_name="save_asset_memory")

    if isinstance(account_memory[0], pd.DataFrame):
        account_df = account_memory[0].copy()
        if "date" in account_df.columns:
            account_df["date"] = pd.to_datetime(account_df["date"])
    else:
        account_df = pd.DataFrame({"date": eval_env.date_memory, "account_value": account_memory[0]})
        account_df["date"] = pd.to_datetime(account_df["date"])

    return account_df, pd.DataFrame(feature_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline RL model directories (A2C/DDPG/TD3/SAC/TQC/PPO)")
    parser.add_argument("--models-root", type=str, required=True, help="For example outputs/fpqc_sac_main/models/sac")
    parser.add_argument("--algorithms", type=str, default="a2c,ddpg,td3,sac,tqc,ppo")
    parser.add_argument("--data-path", type=str, default="data/processed/your_market_panel.csv")
    parser.add_argument("--train-start", type=str, default="2013-01-02")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--test-start", type=str, default="2019-01-02")
    parser.add_argument("--test-end", type=str, default="2021-08-31")
    parser.add_argument("--lookahead-window", type=int, default=5)
    parser.add_argument("--observation-history-window", type=int, default=1,
                        help="Observation history window used for evaluation. Must match training.")
    parser.add_argument("--disable-lookahead", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-simple-mcts",
        action="store_true",
        default=False,
        help="Disable simple MCTS. It is enabled by default to match Simple-MCTS training.",
    )
    parser.add_argument(
        "--simple-mcts-reward-delay-steps",
        type=int,
        default=0,
        help="Delay steps for the MCTS reward component. Must match training.",
    )
    parser.add_argument("--reward-scaling", type=float, default=1e-4)
    parser.add_argument("--cash-scale-factor", type=float, default=1.0,
                        help="Scale only the cash component in observations: state[0] /= this value. Default 1.0.")
    parser.add_argument("--vix-path", type=str, default="data/raw/vix_panel.csv")
    parser.add_argument("--disable-vix", action="store_true")
    parser.add_argument("--vix-threshold", type=float, default=None)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions. Default is deterministic.")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_baseline")
    parser.add_argument("--group-name", type=str, default=None, help="Group name for exports. Defaults to models-root name.")
    parser.add_argument(
        "--export-feature-vectors",
        action="store_true",
        help="Also export encoder feature vectors for each test step.",
    )
    args = parser.parse_args()

    models_root = Path(args.models_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in ALGO_CLASS:
            raise ValueError(f"Unsupported algorithm: {a}")
        if a == "tqc" and TQC is None:
            raise ImportError("TQC requires sb3-contrib.")

    use_vix = not args.disable_vix
    df = mqr.load_and_engineer(args.data_path, use_vix=use_vix, vix_path=args.vix_path, vix_col="vix")
    train_df, test_df = mqr.split_train_test(
        df,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    env_kwargs = mqr.build_env_kwargs(
        train_df,
        lookahead_window=args.lookahead_window,
        reward_scaling=args.reward_scaling,
        use_lookahead_reward=False,
        enable_simple_mcts=not args.disable_simple_mcts,
        simple_mcts_reward_delay_steps=int(args.simple_mcts_reward_delay_steps or 0),
        cash_scale_factor=float(args.cash_scale_factor or 1.0),
        observation_history_window=int(args.observation_history_window or 1),
        use_vix=use_vix,
        vix_threshold=args.vix_threshold,
        vix_col="vix",
    )
    initial_amount = env_kwargs["initial_amount"]

    all_rows: List[Dict] = []
    all_feature_dfs: List[pd.DataFrame] = []
    deterministic = not args.stochastic
    for algo in algos:
        current_group_name = _resolve_group_name(models_root, algo, args.group_name)
        if algo == "ppo":
            model_paths = sorted(
                p for p in models_root.rglob("*.zip") if "ppo" in p.stem.lower()
            )
            if not model_paths:
                print(f"Skipping ppo: no *.zip containing 'ppo' found under {models_root}")
                continue
        else:
            algo_dir = models_root / algo
            model_paths = sorted(algo_dir.glob("*.zip")) if algo_dir.is_dir() else []
            if not model_paths:
                model_paths = sorted(models_root.glob("*.zip"))
            if not model_paths:
                print(f"Skipping {algo}: no model files found at {algo_dir}/*.zip or {models_root}/*.zip")
                continue
        print(f"\nEvaluating {algo.upper()}: {len(model_paths)} models")
        for idx, model_path in enumerate(model_paths, start=1):
            model_name = model_path.stem
            seed = _extract_seed(model_name)
            print(f"[{algo} {idx}/{len(model_paths)}] {model_name}")
            mqr.set_seed(seed)
            model = ALGO_CLASS[algo].load(str(model_path))
            if args.export_feature_vectors:
                account_df, feature_df = _evaluate_once_with_features(
                    model,
                    test_df,
                    env_kwargs,
                    deterministic=deterministic,
                    group_name=current_group_name,
                    algorithm=algo,
                    model_name=model_name,
                    train_seed=seed,
                    eval_seed=seed,
                )
                if len(feature_df) > 0:
                    all_feature_dfs.append(feature_df)
                    feature_csv = output_dir / f"test_feature_vectors_{model_name}.csv"
                    feature_df.to_csv(feature_csv, index=False)
            else:
                account_df = _evaluate_once(model, test_df, env_kwargs, deterministic=deterministic)
            metrics = mqr.compute_metrics(account_df, initial_amount=initial_amount)
            metrics["calmar_ratio"] = _compute_calmar_ratio(
                metrics.get("annual_return", 0.0),
                metrics.get("max_drawdown", 0.0),
            )
            row = {
                "group_name": current_group_name,
                "algorithm": algo,
                "model_name": model_name,
                "seed": seed,
                "m_index": _extract_m_index(model_name),
                **metrics,
            }
            all_rows.append(row)

            account_csv = output_dir / f"test_account_values_{model_name}.csv"
            account_df.to_csv(account_csv, index=False)

    if not all_rows:
        raise RuntimeError("No models were evaluated. Check --models-root.")

    all_df = pd.DataFrame(all_rows)
    all_csv = output_dir / "baseline_all_model_metrics.csv"
    all_df.to_csv(all_csv, index=False)

    if args.export_feature_vectors and all_feature_dfs:
        all_feature_df = pd.concat(all_feature_dfs, ignore_index=True)
        all_feature_csv = output_dir / "baseline_all_model_feature_vectors.csv"
        all_feature_df.to_csv(all_feature_csv, index=False)
        print(f"All-model features: {all_feature_csv}")

    metric_cols = [
        c
        for c in [
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]
        if c in all_df.columns
    ]
    agg = all_df.groupby("algorithm", as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [
        "_".join([x for x in col if x]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    agg_csv = output_dir / "baseline_metrics_by_algorithm_mean_std.csv"
    agg.to_csv(agg_csv, index=False)

    print(f"\nAll-model metrics: {all_csv}")
    print(f"Algorithm mean/std metrics: {agg_csv}")


if __name__ == "__main__":
    main()

