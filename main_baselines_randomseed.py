#!/usr/bin/env python3
"""Train baseline and rule-based strategies for the fixed reproduction workflow."""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


warnings.filterwarnings("ignore", message=".*net_arch.*SB3 v1.8.0.*", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv
try:
    from sb3_contrib import TQC
except Exception:
    TQC = None
import fpqc_sac.repro_utils as mqr


class DiagnosticSAC(SAC):
    """SB3 SAC with additional critic diagnostics logged during training."""

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        td_target_vars, q_overestimation_gaps, q_spreads = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q_cat = torch.cat(current_q_values, dim=1)
            current_q_mean = current_q_cat.mean(dim=1, keepdim=True)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())
            td_target_vars.append(float(torch.var(target_q_values.detach(), unbiased=False).item()))
            q_overestimation_gaps.append(float((current_q_mean.detach() - target_q_values.detach()).mean().item()))
            if current_q_cat.shape[1] > 1:
                q_spreads.append(float((current_q_cat.max(dim=1).values - current_q_cat.min(dim=1).values).mean().item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/critic_loss_var", float(np.var(critic_losses)))
        self.logger.record("train/td_target_var", float(np.mean(td_target_vars)))
        self.logger.record("train/q_overestimation_gap", float(np.mean(q_overestimation_gaps)))
        if q_spreads:
            self.logger.record("train/q_critic_spread", float(np.mean(q_spreads)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))


ALGO_MAP = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": DiagnosticSAC,
    "sac_quantum": DiagnosticSAC,
    "sac_mlp": DiagnosticSAC,
    "sac_linear_bottleneck": DiagnosticSAC,
    "sac_tanh_bottleneck": DiagnosticSAC,
    "sac_clipped_bottleneck": DiagnosticSAC,
    "sac_layernorm_bottleneck": DiagnosticSAC,
    "sac_spectral_bottleneck": DiagnosticSAC,
    "sac_weight_decay_bottleneck": DiagnosticSAC,
    "sac_fourier": DiagnosticSAC,
    "sac_wavelet": DiagnosticSAC,
    "sac_kalman": DiagnosticSAC,
    "tqc": TQC,
}
RULE_ALGOS = {"macd", "kdj_rsi", "zmr", "sma"}
SAC_ALGO_NAMES = {
    "sac",
    "sac_quantum",
    "sac_mlp",
    "sac_linear_bottleneck",
    "sac_tanh_bottleneck",
    "sac_clipped_bottleneck",
    "sac_layernorm_bottleneck",
    "sac_spectral_bottleneck",
    "sac_weight_decay_bottleneck",
    "sac_fourier",
    "sac_wavelet",
    "sac_kalman",
}


def _compute_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return float(annual_return) / (abs(float(max_drawdown)) + 1e-8)


def _parse_net_arch(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        arch = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as e:
        raise ValueError('Invalid argument.') from e
    if not arch or any(width <= 0 for width in arch):
        raise ValueError('Invalid argument.')
    return arch


def _build_baseline_policy_kwargs(algo_name: str, args: argparse.Namespace) -> Optional[Dict]:
    """Build policy kwargs for classic A2C/PPO/DDPG/TD3 baselines."""
    arch = _parse_net_arch(getattr(args, "baseline_net_arch", None))
    if arch is None:
        return None
    algo_name = algo_name.lower()
    if algo_name in {"a2c", "ppo"}:
        net_arch: object = {"pi": arch, "vf": arch}
    elif algo_name in {"ddpg", "td3"}:
        net_arch = arch
    else:
        return None
    print(f"[{algo_name.upper()}] classic policy net_arch={net_arch}")
    return {"net_arch": net_arch}


def _build_sac_policy_kwargs(algo_name: str, args: argparse.Namespace) -> Optional[Dict]:
    algo_name = algo_name.lower()
    if algo_name not in SAC_ALGO_NAMES:
        return None
    sac_net_arch = _parse_net_arch(getattr(args, "sac_net_arch", None))
    if algo_name == "sac" and not getattr(args, "sac_use_quantum_feature", False):
        if sac_net_arch is None:
            return None
        print(f"[{algo_name.upper()}] SAC policy net_arch={sac_net_arch}")
        return {"net_arch": sac_net_arch}

    try:
        import models as custom_models
    except Exception as e:
        raise RuntimeError('Runtime error.') from e

    extractor_class = None
    extractor_kwargs: Dict[str, object] = {"features_dim": 64}
    optimizer_kwargs: Optional[Dict[str, float]] = None
    log_desc = ""

    if algo_name in {"sac", "sac_quantum"}:
        extractor_class = custom_models.QuantumFeatureExtractor
        extractor_kwargs.update(
            {
                "n_qubits": int(args.sac_n_qubits),
                "n_layers": int(getattr(args, "sac_quantum_n_layers", 2)),
                "quantum_device": str(getattr(args, "sac_quantum_device", "cpu")),
                "use_entanglement": bool(getattr(args, "sac_quantum_use_entanglement", True)),
                "entanglement_topology": str(getattr(args, "sac_quantum_entanglement_topology", "ring")),
                "embedding_type": str(getattr(args, "sac_quantum_embedding_type", "angle")),
                "freeze_pqc_params": bool(getattr(args, "sac_quantum_freeze_pqc", False)),
            }
        )
        log_desc = (
            "QuantumFeatureExtractor"
            f"(n_qubits={extractor_kwargs['n_qubits']}, "
            f"n_layers={extractor_kwargs['n_layers']}, "
            f"quantum_device={extractor_kwargs['quantum_device']}, "
            f"use_entanglement={extractor_kwargs['use_entanglement']}, "
            f"entanglement_topology={extractor_kwargs['entanglement_topology']}, "
            f"embedding_type={extractor_kwargs['embedding_type']}, "
            f"freeze_pqc_params={extractor_kwargs['freeze_pqc_params']})"
        )
    elif algo_name == "sac_mlp":
        extractor_class = custom_models.MlpBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"MlpBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_linear_bottleneck":
        extractor_class = custom_models.LinearBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"LinearBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_tanh_bottleneck":
        extractor_class = custom_models.TanhBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"TanhBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_clipped_bottleneck":
        extractor_class = custom_models.ClippedLatentBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"ClippedLatentBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_layernorm_bottleneck":
        extractor_class = custom_models.LayerNormBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"LayerNormBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_spectral_bottleneck":
        extractor_class = custom_models.SpectralNormBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        log_desc = f"SpectralNormBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']})"
    elif algo_name == "sac_weight_decay_bottleneck":
        extractor_class = custom_models.MlpBottleneckExtractor
        extractor_kwargs["bottleneck_dim"] = int(args.sac_mlp_bottleneck_dim)
        optimizer_kwargs = {"weight_decay": float(args.sac_weight_decay)}
        log_desc = (
            f"MlpBottleneckExtractor(bottleneck_dim={extractor_kwargs['bottleneck_dim']}, "
            f"weight_decay={optimizer_kwargs['weight_decay']})"
        )
    elif algo_name == "sac_fourier":
        extractor_class = custom_models.FourierFeatureExtractor
        extractor_kwargs["n_freq"] = int(args.sac_fourier_n_freq)
        log_desc = f"FourierFeatureExtractor(n_freq={extractor_kwargs['n_freq']})"
    elif algo_name == "sac_wavelet":
        extractor_class = custom_models.WaveletExtractor
        extractor_kwargs["init_threshold"] = float(args.sac_wavelet_init_threshold)
        log_desc = f"WaveletExtractor(init_threshold={extractor_kwargs['init_threshold']})"
    elif algo_name == "sac_kalman":
        extractor_class = custom_models.KalmanExtractor
        extractor_kwargs.update(
            {
                "init_q": float(args.sac_kalman_init_q),
                "init_r": float(args.sac_kalman_init_r),
            }
        )
        log_desc = (
            f"KalmanExtractor(init_q={extractor_kwargs['init_q']}, "
            f"init_r={extractor_kwargs['init_r']})"
        )
    else:
        return None

    policy_kwargs: Dict[str, object] = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "share_features_extractor": True,
    }
    if sac_net_arch is not None:
        policy_kwargs["net_arch"] = sac_net_arch
        print(f"[{algo_name.upper()}] SAC policy net_arch={sac_net_arch}")
    if optimizer_kwargs is not None:
        policy_kwargs["optimizer_kwargs"] = optimizer_kwargs
    print(f"[{algo_name.upper()}] feature extractor: {log_desc}")
    return policy_kwargs


def parse_seed_list(args: argparse.Namespace) -> List[int]:
    """Parse fixed, random, or ranged seed settings."""
    if args.fix_seed_list_file:
        path = Path(args.fix_seed_list_file)
        if not path.exists():
            raise FileNotFoundError('File not found.')
        raw = path.read_text(encoding="utf-8").strip()
        seeds: List[int] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.replace(",", " ").split():
                try:
                    seeds.append(int(part))
                except ValueError:
                    pass
        if not seeds:
            raise ValueError('Invalid argument.')
        return seeds

    if args.fix_seed_list:
        parts = [p.strip() for p in args.fix_seed_list.split(",") if p.strip()]
        try:
            seeds = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError('Invalid argument.') from e
        if not seeds:
            raise ValueError('Invalid argument.')
        return seeds

    if args.random_seed_runs and args.random_seed_runs > 0:
        if args.random_seed_master is not None:
            random.seed(args.random_seed_master)
            np.random.seed(args.random_seed_master)
        return [random.randint(0, 2**31 - 1) for _ in range(args.random_seed_runs)]

    # seed: "M" or "M-N"
    seed_arg = str(args.seed).strip()
    if "-" in seed_arg:
        try:
            start_s, count_s = seed_arg.split("-", 1)
            start = int(start_s.strip())
            count = int(count_s.strip())
            return list(range(start, start + count))
        except Exception as e:
            raise ValueError('Invalid argument.') from e
    try:
        return [int(seed_arg)]
    except ValueError as e:
        raise ValueError('Invalid argument.') from e


def save_run_config(args: argparse.Namespace, config_path: str) -> None:
    """Save the run configuration as JSON."""
    keys = [
        "data_path", "algorithms", "timesteps", "checkpoint_every",
        "fix_seed_list_file", "fix_seed_list", "seed", "random_seed_runs", "random_seed_master",
        "n_steps", "reward_log_every", "learn_log_interval", "reward_scaling", "cash_scale_factor", "observation_history_window", "batch_size", "learning_rate", "ent_coef", "buffer_size",
        "sac_use_quantum_feature", "sac_n_qubits", "sac_quantum_n_layers", "sac_quantum_device",
        "sac_quantum_use_entanglement",
        "sac_quantum_entanglement_topology", "sac_quantum_embedding_type", "sac_quantum_freeze_pqc",
        "tqc_n_quantiles", "tqc_top_quantiles_to_drop_per_net",
        "sac_weight_decay", "sac_fourier_n_freq", "sac_mlp_bottleneck_dim",
        "sac_net_arch",
        "baseline_net_arch",
        "sac_use_original_train",
        "sac_wavelet_init_threshold", "sac_kalman_init_q", "sac_kalman_init_r",
        "sma_short_window", "sma_long_window", "macd_fast_window", "macd_slow_window", "macd_signal_window",
        "kdj_window", "kdj_overbought", "kdj_oversold", "rsi_period", "rsi_overbought", "rsi_oversold",
        "zmr_window", "zmr_threshold",
        "train_start", "train_end", "test_start", "test_end", "lookahead_window", "simple_mcts_reward_delay_steps",
        "disable_lookahead", "disable_simple_mcts", "model_index_start",
        "cpu", "cpu_parallel", "serial",
        "vix_path", "disable_vix", "vix_close", "vix_threshold",
        "tensorboard_log", "model_dir", "name_prefix",
    ]
    d: Dict[str, object] = {}
    for k in keys:
        if hasattr(args, k):
            v = getattr(args, k)
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
    
    d["model_type"] = "baseline"
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"Saved run config: {config_path}")


def _require_algo_class(algo_name: str):
    algo_cls = ALGO_MAP.get(algo_name)
    if algo_cls is not None:
        return algo_cls
    if algo_name == "tqc":
        raise ImportError('TQC requires sb3-contrib.')
    raise ValueError('Invalid argument.')


class RewardStatsCallback(BaseCallback):
    """Callback that prints compact reward statistics."""

    def __init__(self, log_every: int = 1024, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_every = max(1, int(log_every))
        self.recent_rewards: List[float] = []
        self.start_time = None
        self.iteration = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            arr = np.asarray(rewards, dtype=float).reshape(-1)
            if arr.size > 0:
                self.recent_rewards.extend(arr.tolist())
                if len(self.recent_rewards) > self.log_every:
                    self.recent_rewards = self.recent_rewards[-self.log_every :]

        if self.num_timesteps % self.log_every == 0:
            self.iteration += 1
            elapsed = max(1e-6, time.time() - (self.start_time or time.time()))
            fps = int(self.num_timesteps / elapsed)
            if self.recent_rewards:
                r = float(self.recent_rewards[-1])
                r_mean = float(np.mean(self.recent_rewards))
                r_min = float(np.min(self.recent_rewards))
                r_max = float(np.max(self.recent_rewards))
            else:
                r = r_mean = r_min = r_max = 0.0

            print("------------------------------------")
            print("| time/              |             |")
            print(f"|    fps             | {fps:<11d}|")
            print(f"|    iterations      | {self.iteration:<11d}|")
            print(f"|    time_elapsed    | {int(elapsed):<11d}|")
            print(f"|    total_timesteps | {self.num_timesteps:<11d}|")
            print("| train/             |             |")
            print(f"|    reward          | {r:<11.6f}|")
            print(f"|    reward_max      | {r_max:<11.6f}|")
            print(f"|    reward_mean     | {r_mean:<11.6f}|")
            print(f"|    reward_min      | {r_min:<11.6f}|")
            print("------------------------------------")
        return True


def train_one(
    algo_name: str,
    train_df,
    env_kwargs: Dict,
    total_timesteps: int,
    model_path: str,
    tensorboard_log: str,
    seed: int,
    args: argparse.Namespace,
) -> None:
    """Train one baseline algorithm for one seed."""
    algo_name = algo_name.lower()
    if algo_name not in ALGO_MAP:
        raise ValueError('Invalid argument.')
    algo_cls = _require_algo_class(algo_name)
    if algo_name in SAC_ALGO_NAMES and bool(getattr(args, "sac_use_original_train", False)):
        algo_cls = SAC
        print(f"[{algo_name.upper()}] using SB3 native SAC.train().")

    mqr.set_seed(seed)
    vec_env = DummyVecEnv([lambda: mqr.make_env(train_df, env_kwargs)])
    device = "cpu" if args.cpu else "auto"

    common_kwargs = dict(
        env=vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        device=device,
        learning_rate=args.learning_rate,
    )

    
    if algo_name == "a2c":
        policy_kwargs = _build_baseline_policy_kwargs(algo_name, args)
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
        )
    elif algo_name == "ppo":
        policy_kwargs = _build_baseline_policy_kwargs(algo_name, args)
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
        )
    elif algo_name == "ddpg":
        policy_kwargs = _build_baseline_policy_kwargs(algo_name, args)
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
        )
    elif algo_name == "td3":
        policy_kwargs = _build_baseline_policy_kwargs(algo_name, args)
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
        )
    elif algo_name in SAC_ALGO_NAMES:
        policy_kwargs = _build_sac_policy_kwargs(algo_name, args)
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
        )
    elif algo_name == "tqc":
        policy_kwargs: Dict[str, object] = {"n_quantiles": int(args.tqc_n_quantiles)}
        model = algo_cls(
            "MlpPolicy",
            **common_kwargs,
            policy_kwargs=policy_kwargs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            top_quantiles_to_drop_per_net=int(args.tqc_top_quantiles_to_drop_per_net),
        )
    else:
        raise ValueError('Invalid argument.')

    callbacks_list = [RewardStatsCallback(log_every=args.reward_log_every)]
    if args.checkpoint_every and args.checkpoint_every > 0:
        ckpt_dir = os.path.join(args.model_dir, f"checkpoints_{Path(model_path).stem}")
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks_list.append(CheckpointCallback(save_freq=args.checkpoint_every, save_path=ckpt_dir, name_prefix="step"))
        print(f"Saving checkpoints every {args.checkpoint_every} steps to: {ckpt_dir}")
        save_run_config(args, os.path.join(ckpt_dir, "run_config.json"))

    print(f"Starting {algo_name.upper()} training (seed={seed})...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks_list,
        log_interval=int(getattr(args, "learn_log_interval", 4)),
    )
    model.save(model_path)
    print(f"Saved model: {model_path}")
    vec_env.close()


def parse_algorithms(s: str) -> List[str]:
    items = [x.strip().lower() for x in s.replace(";", ",").split(",") if x.strip()]
    if not items:
        raise ValueError('Invalid argument.')
    for a in items:
        if a not in ALGO_MAP and a not in RULE_ALGOS:
            raise ValueError('Unsupported algorithm.')
    return items


def build_jobs(algorithms: List[str], seeds: List[int]) -> List[Tuple[str, int, int]]:
    jobs: List[Tuple[str, int, int]] = []
    idx = 0
    for algo in algorithms:
        for seed in seeds:
            jobs.append((algo, seed, idx))
            idx += 1
    return jobs


def _build_child_cmd(
    args: argparse.Namespace,
    script_path: Path,
    algo: str,
    seed: int,
    model_dir: str,
    tb_dir: str,
    model_index_start: int = 0,
) -> List[str]:
    """Build a child-process command for one algorithm and seed."""
    cmd = [
        args.python_exe,
        str(script_path),
        "--data-path",
        args.data_path,
        "--algorithms",
        algo,
        "--seed",
        str(seed),
        "--timesteps",
        str(args.timesteps),
        "--train-start",
        args.train_start,
        "--train-end",
        args.train_end,
        "--test-start",
        args.test_start,
        "--test-end",
        args.test_end,
        "--lookahead-window",
        str(args.lookahead_window),
        "--simple-mcts-reward-delay-steps",
        str(getattr(args, "simple_mcts_reward_delay_steps", 0)),
        "--observation-history-window",
        str(getattr(args, "observation_history_window", 1)),
        "--reward-scaling",
        str(args.reward_scaling),
        "--cash-scale-factor",
        str(getattr(args, "cash_scale_factor", 1.0)),
        "--learning-rate",
        str(args.learning_rate),
        "--ent-coef",
        str(args.ent_coef),
        "--batch-size",
        str(args.batch_size),
        "--buffer-size",
        str(args.buffer_size),
        "--n-steps",
        str(args.n_steps),
        "--reward-log-every",
        str(args.reward_log_every),
        "--learn-log-interval",
        str(getattr(args, "learn_log_interval", 4)),
        "--sma-short-window",
        str(args.sma_short_window),
        "--sma-long-window",
        str(args.sma_long_window),
        "--macd-fast-window",
        str(args.macd_fast_window),
        "--macd-slow-window",
        str(args.macd_slow_window),
        "--macd-signal-window",
        str(args.macd_signal_window),
        "--kdj-window",
        str(args.kdj_window),
        "--kdj-overbought",
        str(args.kdj_overbought),
        "--kdj-oversold",
        str(args.kdj_oversold),
        "--rsi-period",
        str(args.rsi_period),
        "--rsi-overbought",
        str(args.rsi_overbought),
        "--rsi-oversold",
        str(args.rsi_oversold),
        "--zmr-window",
        str(args.zmr_window),
        "--zmr-threshold",
        str(args.zmr_threshold),
        "--model-dir",
        model_dir,
        "--tensorboard-log",
        tb_dir,
        "--name-prefix",
        args.name_prefix,
        "--parallel-workers",
        "1",
        "--child-run",
        "--run-tag",
        args.run_tag,
        "--python-exe",
        args.python_exe,
        "--model-index-start",
        str(int(model_index_start)),
    ]
    if args.checkpoint_every:
        cmd += ["--checkpoint-every", str(args.checkpoint_every)]
    if args.disable_lookahead:
        cmd.append("--disable-lookahead")
    if args.disable_simple_mcts:
        cmd.append("--disable-simple-mcts")
    if args.cpu:
        cmd.append("--cpu")
    if args.disable_vix:
        cmd.append("--disable-vix")
    if args.vix_close:
        cmd.append("--vix-close")
    if args.vix_threshold is not None:
        cmd += ["--vix-threshold", str(args.vix_threshold)]
    if args.vix_path:
        cmd += ["--vix-path", args.vix_path]
    cmd += [
        "--sac-n-qubits",
        str(getattr(args, "sac_n_qubits", 7)),
        "--sac-quantum-n-layers",
        str(getattr(args, "sac_quantum_n_layers", 2)),
        "--sac-quantum-device",
        str(getattr(args, "sac_quantum_device", "cpu")),
        "--sac-quantum-entanglement-topology",
        str(getattr(args, "sac_quantum_entanglement_topology", "ring")),
        "--sac-quantum-embedding-type",
        str(getattr(args, "sac_quantum_embedding_type", "angle")),
        "--sac-net-arch",
        str(getattr(args, "sac_net_arch", "") or ""),
        "--baseline-net-arch",
        str(getattr(args, "baseline_net_arch", "") or ""),
        "--sac-mlp-bottleneck-dim",
        str(getattr(args, "sac_mlp_bottleneck_dim", 7)),
        "--sac-fourier-n-freq",
        str(getattr(args, "sac_fourier_n_freq", 7)),
        "--sac-weight-decay",
        str(getattr(args, "sac_weight_decay", 1e-4)),
        "--sac-wavelet-init-threshold",
        str(getattr(args, "sac_wavelet_init_threshold", 0.1)),
        "--sac-kalman-init-q",
        str(getattr(args, "sac_kalman_init_q", 1.0)),
        "--sac-kalman-init-r",
        str(getattr(args, "sac_kalman_init_r", 1.0)),
        "--tqc-n-quantiles",
        str(getattr(args, "tqc_n_quantiles", 25)),
        "--tqc-top-quantiles-to-drop-per-net",
        str(getattr(args, "tqc_top_quantiles_to_drop_per_net", 2)),
    ]
    if getattr(args, "sac_use_quantum_feature", False):
        cmd.append("--sac-use-quantum-feature")
    if getattr(args, "sac_quantum_freeze_pqc", False):
        cmd.append("--sac-quantum-freeze-pqc")
    if not getattr(args, "sac_quantum_use_entanglement", True):
        cmd.append("--sac-quantum-no-entanglement")
    if getattr(args, "sac_use_original_train", False):
        cmd.append("--sac-use-original-train")
    return cmd


def run_parallel_rl_jobs(args: argparse.Namespace, algorithms: List[str], seeds: List[int]) -> None:
    """Run RL training jobs with a bounded worker pool."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    rl_algos = [a for a in algorithms if a in ALGO_MAP]
    skipped = [a for a in algorithms if a not in ALGO_MAP]
    if skipped:
        print(f"Parallel mode only schedules RL algorithms. Skipped: {skipped}")

    tb_path = Path(args.tensorboard_log)
    if tb_path.name == "tb":
        default_tb_dir = tb_path
        default_log_dir = tb_path.parent
    else:
        default_log_dir = None
        default_tb_dir = None

    jobs = []
    m_cursor = int(getattr(args, "model_index_start", 0) or 0)
    for algo in rl_algos:
        md = Path(args.model_dir)
        if md.name == algo:
            model_dir = md
        else:
            model_dir = project_root / f"trained_models_{args.run_tag}" / algo
        if default_log_dir is not None:
            log_dir = default_log_dir
            tb_dir = default_tb_dir
        else:
            log_dir = project_root / f"logs_{args.run_tag}" / algo
            tb_dir = log_dir / "tb"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            log_file = log_dir / f"{args.name_prefix}_{algo}_s{seed}_train.log"
            cmd = _build_child_cmd(
                args, script_path, algo, seed, str(model_dir), str(tb_dir), model_index_start=m_cursor
            )
            jobs.append((algo, seed, cmd, log_file))
            m_cursor += 1

    if not jobs:
        print("No RL jobs to run in parallel.")
        return

    max_workers = max(1, int(args.parallel_workers))
    print(f"Starting parallel training: jobs={len(jobs)}, workers={max_workers}")

    pending = list(jobs)
    running = []
    success = 0
    failed = 0

    while pending or running:
        while pending and len(running) < max_workers:
            algo, seed, cmd, log_file = pending.pop(0)
            fh = open(log_file, "w", encoding="utf-8")
            fh.write("# command:\n")
            fh.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
            fh.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            running.append((algo, seed, proc, fh, log_file))
            print(f"[start] {algo} seed={seed} -> {log_file}")

        new_running = []
        for algo, seed, proc, fh, log_file in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((algo, seed, proc, fh, log_file))
                continue
            fh.close()
            if ret == 0:
                success += 1
                print(f"[done] {algo} seed={seed}")
            else:
                failed += 1
                print(f"[failed] {algo} seed={seed}, exit={ret}, log={log_file}")
        running = new_running
        if running:
            time.sleep(1.0)

    print(f"Parallel training finished: success={success}, failed={failed}")
    if failed > 0:
        raise SystemExit(1)


def _ensure_indicator_columns(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Add indicators used by rule-based baselines."""
    out = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    grp = out.groupby("tic", group_keys=False)

    
    out["sma_short"] = grp["close"].transform(
        lambda s: s.rolling(args.sma_short_window, min_periods=args.sma_short_window).mean()
    )
    out["sma_long"] = grp["close"].transform(
        lambda s: s.rolling(args.sma_long_window, min_periods=args.sma_long_window).mean()
    )
    out["sma_short_prev"] = grp["sma_short"].shift(1)
    out["sma_long_prev"] = grp["sma_long"].shift(1)

    
    close_diff = grp["close"].diff()
    gain = close_diff.clip(lower=0.0)
    loss = (-close_diff).clip(lower=0.0)
    avg_gain = gain.groupby(out["tic"]).transform(
        lambda s: s.rolling(args.rsi_period, min_periods=args.rsi_period).mean()
    )
    avg_loss = loss.groupby(out["tic"]).transform(
        lambda s: s.rolling(args.rsi_period, min_periods=args.rsi_period).mean()
    )
    rs = avg_gain / (avg_loss + 1e-8)
    out["rsi_use"] = 100.0 - (100.0 / (1.0 + rs))

    
    ema_fast = grp["close"].transform(lambda s: s.ewm(span=args.macd_fast_window, adjust=False).mean())
    ema_slow = grp["close"].transform(lambda s: s.ewm(span=args.macd_slow_window, adjust=False).mean())
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].groupby(out["tic"]).transform(
        lambda s: s.ewm(span=args.macd_signal_window, adjust=False).mean()
    )
    out["macd_prev"] = grp["macd"].shift(1)
    out["macd_signal_prev"] = grp["macd_signal"].shift(1)

    
    out["zmr_mean"] = grp["close"].transform(
        lambda s: s.rolling(args.zmr_window, min_periods=args.zmr_window).mean()
    )
    out["zmr_dev"] = (out["close"] - out["zmr_mean"]) / (out["zmr_mean"] + 1e-8)  
    out["zmr_dev_prev"] = grp["zmr_dev"].shift(1)

    # KDJ
    low_n = grp["low"].transform(lambda s: s.rolling(args.kdj_window, min_periods=args.kdj_window).min())
    high_n = grp["high"].transform(lambda s: s.rolling(args.kdj_window, min_periods=args.kdj_window).max())
    rsv = (out["close"] - low_n) / (high_n - low_n + 1e-8) * 100.0
    out["kdj_k"] = rsv.groupby(out["tic"]).transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    out["kdj_d"] = out["kdj_k"].groupby(out["tic"]).transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    out["kdj_k_prev"] = grp["kdj_k"].shift(1)
    out["kdj_d_prev"] = grp["kdj_d"].shift(1)
    return out


def _rule_signal(df: pd.DataFrame, algo_name: str, args: argparse.Namespace) -> pd.Series:
    """Generate daily rule-based trading signals."""
    algo = algo_name.lower()
    sig = pd.Series(0.0, index=df.index)
    if algo == "sma":
        buy = (df["sma_short"] > df["sma_long"]) & (df["sma_short_prev"] <= df["sma_long_prev"])
        sell = (df["sma_short"] < df["sma_long"]) & (df["sma_short_prev"] >= df["sma_long_prev"])
        sig[buy] = 1.0
        sig[sell] = -1.0
    elif algo == "macd":
        buy = (df["macd"] > df["macd_signal"]) & (df["macd_prev"] <= df["macd_signal_prev"])
        sell = (df["macd"] < df["macd_signal"]) & (df["macd_prev"] >= df["macd_signal_prev"])
        sig[buy] = 1.0
        sig[sell] = -1.0
    elif algo == "zmr":
        
        buy = (df["zmr_dev_prev"] < -args.zmr_threshold) & (df["zmr_dev"] >= 0)
        sell = (df["zmr_dev_prev"] > args.zmr_threshold) & (df["zmr_dev"] <= 0)
        sig[buy] = 1.0
        sig[sell] = -1.0
    elif algo == "kdj_rsi":
        
        buy = (
            (df["kdj_k"] > df["kdj_d"])
            & (df["kdj_k_prev"] <= df["kdj_d_prev"])
            & (df["kdj_k"] < args.kdj_oversold)
            & (df["rsi_use"] < args.rsi_oversold)
        )
        sell = (
            (df["kdj_k"] < df["kdj_d"])
            & (df["kdj_k_prev"] >= df["kdj_d_prev"])
            & (df["kdj_k"] > args.kdj_overbought)
            & (df["rsi_use"] > args.rsi_overbought)
        )
        sig[buy] = 1.0
        sig[sell] = -1.0
    else:
        raise ValueError('Invalid argument.')
    return sig.fillna(0.0)


def _extract_obs(reset_ret):
    if isinstance(reset_ret, tuple):
        return reset_ret[0]
    return reset_ret


def _rule_params_for_algo(algo_name: str, args: argparse.Namespace) -> Dict[str, object]:
    """Collect rule-based strategy parameters for output files."""
    algo = algo_name.lower()
    common = {
        "test_start": args.test_start,
        "test_end": args.test_end,
        "disable_lookahead": bool(args.disable_lookahead),
        "disable_simple_mcts": bool(args.disable_simple_mcts),
        "reward_scaling": float(args.reward_scaling),
    }
    if algo == "macd":
        common.update(
            {
                "macd_fast_window": int(args.macd_fast_window),
                "macd_slow_window": int(args.macd_slow_window),
                "macd_signal_window": int(args.macd_signal_window),
            }
        )
    elif algo == "sma":
        common.update(
            {
                "sma_short_window": int(args.sma_short_window),
                "sma_long_window": int(args.sma_long_window),
            }
        )
    elif algo == "kdj_rsi":
        common.update(
            {
                "kdj_window": int(args.kdj_window),
                "kdj_overbought": float(args.kdj_overbought),
                "kdj_oversold": float(args.kdj_oversold),
                "rsi_period": int(args.rsi_period),
                "rsi_overbought": float(args.rsi_overbought),
                "rsi_oversold": float(args.rsi_oversold),
            }
        )
    elif algo == "zmr":
        common.update(
            {
                "zmr_window": int(args.zmr_window),
                "zmr_threshold": float(args.zmr_threshold),
            }
        )
    return common


def _run_rule_backtest(
    algo_name: str,
    test_df: pd.DataFrame,
    env_kwargs: Dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run one deterministic rule-based backtest."""
    if algo_name.lower() == "buy_hold":
        raise ValueError('Invalid argument.')

    out_df = _ensure_indicator_columns(test_df, args=args)
    out_df["rule_signal"] = _rule_signal(out_df, algo_name, args=args)
    out_df = out_df.sort_values(["date", "tic"]).copy()
    out_df.index = out_df["date"].factorize()[0]

    env = mqr.make_env(out_df, env_kwargs)
    _obs = _extract_obs(env.reset())
    done, truncated = False, False

    while not done and not truncated:
        day_slice = out_df.loc[env.day, :]
        if isinstance(day_slice, pd.Series):
            day_slice = day_slice.to_frame().T

        
        if isinstance(env.data, pd.DataFrame):
            env_tics = list(env.data["tic"].values)
        else:
            env_tics = [str(env.data.get("tic", day_slice["tic"].iloc[0]))]
        sig_map = day_slice.set_index("tic")["rule_signal"].to_dict()
        actions = np.array([float(sig_map.get(t, 0.0)) for t in env_tics], dtype=np.float32)

        step_ret = env.step(actions)
        if isinstance(step_ret, tuple) and len(step_ret) == 5:
            _obs, _reward, done, truncated, _info = step_ret
        elif isinstance(step_ret, tuple) and len(step_ret) == 4:
            _obs, _reward, done, _info = step_ret
            truncated = False
        else:
            break

    dates = pd.to_datetime(env.date_memory)
    values = np.asarray(env.asset_memory, dtype=float)
    n = min(len(dates), len(values))
    account_df = pd.DataFrame({"date": dates[:n], "account_value": values[:n]})
    metrics = mqr.compute_metrics(account_df, initial_amount=env_kwargs["initial_amount"])
    metrics["calmar_ratio"] = _compute_calmar_ratio(
        metrics.get("annual_return", 0.0),
        metrics.get("max_drawdown", 0.0),
    )
    metrics_payload = dict(metrics)
    metrics_payload.update(_rule_params_for_algo(algo_name, args))
    metrics_payload["algorithm"] = algo_name.lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    account_csv = output_dir / f"rule_{algo_name}_account_values.csv"
    metrics_csv = output_dir / f"rule_{algo_name}_metrics.csv"
    account_df.to_csv(account_csv, index=False)
    pd.DataFrame([metrics_payload]).to_csv(metrics_csv, index=False)
    print(f"Saved rule-based[{algo_name}] account curve: {account_csv}")
    print(f"Saved rule-based[{algo_name}] metrics: {metrics_csv}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description='Train and evaluate the configured strategies.')
    parser.add_argument("--data-path", type=str, default="data/processed/your_market_panel.csv")
    parser.add_argument("--algorithms", type=str, default="a2c,ddpg,td3,sac,macd,kdj_rsi,zmr,sma",
                        help='Comma-separated list of algorithms to run.')
    parser.add_argument("--timesteps", type=int, default=25000)
    parser.add_argument("--checkpoint-every", type=int, default=25000,
                        help='Save a checkpoint every N training steps.')

    parser.add_argument("--seed", type=str, default="42",
                        help="Seed setting, either 'M' or 'M-N'.")
    parser.add_argument("--random-seed-runs", type=int, default=0,
                        help='Number of random seed runs to generate.')
    parser.add_argument("--random-seed-master", type=int, default=None,
                        help='Master seed used to generate random seeds.')
    parser.add_argument("--fix-seed-list", type=str, default=None,
                        help='Comma-separated fixed seed list.')
    parser.add_argument("--fix-seed-list-file", type=str, default=None,
                        help='Path to a fixed seed list file.')

    parser.add_argument("--train-start", type=str, default="2013-01-02")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--test-start", type=str, default="2019-01-02")
    parser.add_argument("--test-end", type=str, default="2021-08-31")
    parser.add_argument("--lookahead-window", type=int, default=5)
    parser.add_argument("--observation-history-window", type=int, default=1,
                        help='Number of observation steps concatenated for the agent.')
    parser.add_argument(
        "--simple-mcts-reward-delay-steps",
        type=int,
        default=10,
        help='Delay the environment simple-MCTS reward component by N trading days.',
    )
    parser.add_argument("--disable-lookahead", action="store_true",
                        default=True,
                        help='Disable the look-ahead reward and use one-step asset change.')
    parser.add_argument("--disable-simple-mcts", action="store_true",
                        help='Disable the environment simple-MCTS reward component.')
    parser.add_argument("--reward-scaling", type=float, default=1e-4)
    parser.add_argument("--cash-scale-factor", type=float, default=1.0,
                        help='Scale the cash component in observations by this factor.')

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help='Entropy coefficient.')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Mini-batch size.')
    parser.add_argument("--buffer-size", type=int, default=200000,
                        help='Replay buffer size.')
    parser.add_argument("--n-steps", type=int, default=1024,
                        help='Number of rollout steps per update.')
    parser.add_argument(
        "--sac-n-qubits",
        dest="sac_n_qubits",
        type=int,
        default=7,
        help='Number of qubits for FPQC-SAC.',
    )
    parser.add_argument(
        "--sac-quantum-n-layers",
        dest="sac_quantum_n_layers",
        type=int,
        default=2,
        help='Number of PQC layers for FPQC-SAC.',
    )
    parser.add_argument(
        "--sac-quantum-device",
        dest="sac_quantum_device",
        type=str,
        default="cpu",
        help='Device used by the FPQC-SAC quantum layer.',
    )
    parser.add_argument("--sac-use-quantum-feature", action="store_true",
                        help='Use the FPQC feature extractor for SAC.')
    parser.add_argument(
        "--sac-quantum-no-entanglement",
        dest="sac_quantum_use_entanglement",
        action="store_false",
        default=True,
        help='Disable entanglement in the SAC quantum feature extractor.',
    )
    parser.add_argument(
        "--sac-quantum-entanglement-topology",
        type=str,
        default="ring",
        choices=["ring", "line"],
        help='Entanglement topology for the SAC quantum feature extractor.',
    )
    parser.add_argument(
        "--sac-quantum-embedding-type",
        type=str,
        default="angle",
        choices=["angle", "amplitude"],
        help='Embedding type for the SAC quantum feature extractor.',
    )
    parser.add_argument(
        "--sac-quantum-freeze-pqc",
        action="store_true",
        default=False,
        help='Freeze PQC parameters in the SAC quantum feature extractor.',
    )
    parser.add_argument(
        "--sac-net-arch",
        type=str,
        default="",
        help='Comma-separated SAC network architecture, for example 256,256.',
    )
    parser.add_argument(
        "--baseline-net-arch",
        type=str,
        default="",
        help='Comma-separated classic baseline network architecture.',
    )
    parser.add_argument(
        "--sac-use-original-train",
        action="store_true",
        default=False,
        help='Use the original SB3 SAC train method instead of diagnostic SAC.',
    )
    parser.add_argument("--sac-mlp-bottleneck-dim", type=int, default=7,
                        help='Bottleneck dimension for the SAC MLP encoder.')
    parser.add_argument("--sac-fourier-n-freq", type=int, default=7,
                        help='Number of Fourier frequencies for the SAC Fourier encoder.')
    parser.add_argument("--sac-weight-decay", type=float, default=1e-4,
                        help='Weight decay for the SAC weight-decay bottleneck.')
    parser.add_argument("--sac-wavelet-init-threshold", type=float, default=0.1,
                        help='Initial soft-threshold value for the SAC wavelet encoder.')
    parser.add_argument("--sac-kalman-init-q", type=float, default=1.0,
                        help='Initial process-noise value for the SAC Kalman encoder.')
    parser.add_argument("--sac-kalman-init-r", type=float, default=1.0,
                        help='Initial measurement-noise value for the SAC Kalman encoder.')
    parser.add_argument("--tqc-n-quantiles", type=int, default=25,
                        help='Number of quantiles for the TQC baseline.')
    parser.add_argument("--tqc-top-quantiles-to-drop-per-net", type=int, default=2,
                        help='Number of top quantiles to drop per TQC network.')
    parser.add_argument("--reward-log-every", type=int, default=1024,
                        help='Log reward statistics every N steps.')
    parser.add_argument("--learn-log-interval", type=int, default=4,
                        help='SB3 learning log interval.')

    
    parser.add_argument("--sma-short-window", type=int, default=60, help='Short moving-average window for the SMA rule.')
    parser.add_argument("--sma-long-window", type=int, default=80, help='Long moving-average window for the SMA rule.')
    parser.add_argument("--macd-fast-window", type=int, default=3, help='Fast EMA window for MACD.')
    parser.add_argument("--macd-slow-window", type=int, default=10, help='Slow EMA window for MACD.')
    parser.add_argument("--macd-signal-window", type=int, default=3, help='Signal EMA window for MACD.')
    parser.add_argument("--kdj-window", type=int, default=14, help='Lookback window for KDJ.')
    parser.add_argument("--kdj-overbought", type=float, default=90.0, help='KDJ overbought threshold.')
    parser.add_argument("--kdj-oversold", type=float, default=20.0, help='KDJ oversold threshold.')
    parser.add_argument("--rsi-period", type=int, default=14, help='RSI lookback period.')
    parser.add_argument("--rsi-overbought", type=float, default=80.0, help='RSI overbought threshold.')
    parser.add_argument("--rsi-oversold", type=float, default=20.0, help='RSI oversold threshold.')
    parser.add_argument("--zmr-window", type=int, default=3, help='Rolling mean window for the ZMR rule.')
    parser.add_argument("--zmr-threshold", type=float, default=0.001, help='Deviation threshold for the ZMR rule.')

    parser.add_argument("--cpu", action="store_true", help='Force CPU execution.')
    parser.add_argument("--cpu-parallel", type=int, default=1,
                        help='Number of CPU workers for parallel runs.')
    parser.add_argument("--serial", action="store_true", help='Run jobs serially instead of using the parallel scheduler.')

    parser.add_argument("--vix-path", type=str, default="data/raw/vix_panel.csv")
    parser.add_argument("--disable-vix", action="store_true")
    parser.add_argument("--vix-close", action="store_true")
    parser.add_argument("--vix-threshold", type=float, default=None)

    parser.add_argument("--tensorboard-log", type=str, default="logs")
    parser.add_argument("--model-dir", type=str, default="trained_models",
                        help='Directory for saved baseline models.')
    parser.add_argument("--rule-output-dir", type=str, default="results/rule_based_baselines",
                        help='Directory for rule-based baseline outputs.')
    parser.add_argument("--name-prefix", type=str, default="baseline",
                        help='Prefix used in saved model names.')
    parser.add_argument(
        "--model-index-start",
        type=int,
        default=0,
        help='Starting model index used in saved names.',
    )
    parser.add_argument("--parallel-workers", type=int, default=1,
                        help='Maximum number of parallel worker processes.')
    parser.add_argument("--child-run", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--run-tag", type=str, default="baseline_run",
                        help='Tag used to group output folders.')
    parser.add_argument("--python-exe", type=str, default=sys.executable,
                        help='Python executable used for child processes.')
    args = parser.parse_args()
    if args.macd_fast_window >= args.macd_slow_window:
        raise ValueError('Invalid argument.')
    if args.tqc_n_quantiles <= 0:
        raise ValueError('Invalid argument.')
    if args.tqc_top_quantiles_to_drop_per_net < 0 or args.tqc_top_quantiles_to_drop_per_net >= args.tqc_n_quantiles:
        raise ValueError('Invalid argument.')

    algorithms = parse_algorithms(args.algorithms)
    seeds = parse_seed_list(args)
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {seeds}")

    
    
    if (not args.child_run) and args.parallel_workers >= 1 and all(a in ALGO_MAP for a in algorithms):
        run_parallel_rl_jobs(args, algorithms, seeds)
        return

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    use_vix = not (args.disable_vix or args.vix_close)
    vix_col = "vix"
    if use_vix:
        vix_path = Path(args.vix_path)
        if not vix_path.exists():
            raise FileNotFoundError('File not found.')

    
    mqr.set_seed(seeds[0])

    print("=" * 60)
    print("Loading and preprocessing data")
    print("=" * 60)
    df = mqr.load_and_engineer(
        args.data_path,
        use_vix=use_vix,
        vix_path=args.vix_path,
        vix_col=vix_col,
    )
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
        use_lookahead_reward=not args.disable_lookahead,
        enable_simple_mcts=not args.disable_simple_mcts,
        simple_mcts_reward_delay_steps=getattr(args, "simple_mcts_reward_delay_steps", 0) or 0,
        cash_scale_factor=getattr(args, "cash_scale_factor", 1.0) or 1.0,
        observation_history_window=getattr(args, "observation_history_window", 1) or 1,
        use_vix=use_vix,
        vix_threshold=args.vix_threshold,
        vix_col=vix_col,
    )

    jobs = build_jobs(algorithms, seeds)
    print(f"Total scheduled jobs: {len(jobs)}")

    rule_done: Set[str] = set()
    rule_metrics_rows: List[Dict[str, object]] = []
    for algo, seed, idx in jobs:
        if algo in RULE_ALGOS:
            if algo in rule_done:
                continue
            print("\n" + "=" * 60)
            print(f"Running rule-based strategy: {algo}")
            print("=" * 60)
            rb_out = Path(args.rule_output_dir)
            metrics = _run_rule_backtest(
                algo_name=algo,
                test_df=test_df,
                env_kwargs=env_kwargs,
                output_dir=rb_out,
                args=args,
            )
            row = {"algorithm": algo, "seed": "deterministic"}
            row.update(metrics)
            rule_metrics_rows.append(row)
            rule_done.add(algo)
            continue

        name_idx = idx + int(getattr(args, "model_index_start", 0) or 0)
        model_name = f"{args.name_prefix}_{algo}_s{seed}_m{name_idx}"
        model_path = f"{args.model_dir}/{model_name}.zip"
        tb_log = os.path.join(args.tensorboard_log, model_name)

        print("\n" + "=" * 60)
        print(f"Training {algo.upper()} (seed={seed})")
        print("=" * 60)
        save_run_config(args, f"{args.model_dir}/{model_name}_run_config.json")
        train_one(
            algo_name=algo,
            train_df=train_df,
            env_kwargs=env_kwargs,
            total_timesteps=args.timesteps,
            model_path=model_path,
            tensorboard_log=tb_log,
            seed=seed,
            args=args,
        )

    if rule_metrics_rows:
        rb_out = Path(args.rule_output_dir)
        summary_csv = rb_out / "rule_based_metrics_summary.csv"
        pd.DataFrame(rule_metrics_rows).to_csv(summary_csv, index=False)
        print(f"Saved rule-based metrics summary: {summary_csv}")

    print("All baseline jobs finished.")
    print("Outputs are saved under the configured model, log, result, and rule output directories.")


if __name__ == "__main__":
    main()

