"""Configuration-based launcher for FPQC-SAC experiments.

The runner intentionally keeps the existing research training code intact and
wraps it with a small, auditable layer that maps YAML configuration files to
the legacy command-line interfaces.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RL_ALGOS = {
    "a2c",
    "ppo",
    "ddpg",
    "td3",
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
    "tqc",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON experiment configuration."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "YAML configs require PyYAML. Install dependencies with "
            "`pip install -r requirements.txt`, or use a JSON config."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_dicts(*items: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge shallow dictionaries, skipping null inputs."""
    out: dict[str, Any] = {}
    for item in items:
        if item:
            out.update(dict(item))
    return out


def as_list(value: Any) -> list[str]:
    """Normalize comma-separated strings and sequences to a string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def relpath(path: Path) -> str:
    """Return a stable path relative to the repository root when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(raw: str | Path) -> Path:
    """Resolve paths relative to the repository root."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def cli_key(key: str) -> str:
    """Convert a config key to a CLI flag."""
    return "--" + key.replace("_", "-")


def append_cli_args(cmd: list[str], values: Mapping[str, Any]) -> list[str]:
    """Append argparse-style flags from a mapping.

    Boolean true values become standalone flags. Boolean false and null values
    are omitted because the legacy scripts use action flags for negation.
    """
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = cli_key(key)
        if value is True:
            cmd.append(flag)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            cmd.extend([flag, ",".join(str(x) for x in value)])
        else:
            cmd.extend([flag, str(value)])
    return cmd


def experiment_output_dir(output_root: Path, experiment_id: str) -> Path:
    """Return the canonical output directory for an experiment."""
    return output_root / experiment_id


def build_train_commands(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    python_exe: str,
) -> list[list[str]]:
    """Build one or more training commands for an experiment."""
    defaults = config.get("defaults", {})
    default_train = defaults.get("train", {}) if isinstance(defaults, Mapping) else {}
    train_args = merge_dicts(default_train, experiment.get("train", {}))
    exp_id = str(experiment["id"])
    output_root = resolve_path(str(config.get("output_root", "outputs")))
    exp_out = experiment_output_dir(output_root, exp_id)

    algorithms = as_list(train_args.get("algorithms", "sac"))
    if not algorithms:
        raise ValueError(f"Experiment {exp_id} has no algorithms")

    commands = []
    for offset, algo in enumerate(algorithms):
        per_algo_args = dict(train_args)
        per_algo_args["algorithms"] = algo
        per_algo_args.setdefault("run_tag", exp_id)
        per_algo_args.setdefault("name_prefix", "baseline")
        per_algo_args.setdefault("model_index_start", offset)
        if algo in RL_ALGOS:
            per_algo_args["model_dir"] = relpath(exp_out / "models" / algo)
            per_algo_args["tensorboard_log"] = relpath(exp_out / "logs" / algo / "tb")
        else:
            per_algo_args["rule_output_dir"] = relpath(exp_out / "eval" / "rule_based")

        cmd = [python_exe, "main_baselines_randomseed.py"]
        append_cli_args(cmd, per_algo_args)
        commands.append(cmd)
    return commands


def build_eval_commands(
    config: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    python_exe: str,
) -> list[list[str]]:
    """Build evaluation commands for an experiment."""
    defaults = config.get("defaults", {})
    default_eval = defaults.get("eval", {}) if isinstance(defaults, Mapping) else {}
    train_args = merge_dicts(defaults.get("train", {}) if isinstance(defaults, Mapping) else {}, experiment.get("train", {}))
    eval_args = merge_dicts(default_eval, experiment.get("eval", {}))
    exp_id = str(experiment["id"])
    output_root = resolve_path(str(config.get("output_root", "outputs")))
    exp_out = experiment_output_dir(output_root, exp_id)

    algorithms = as_list(eval_args.get("algorithms") or train_args.get("algorithms", "sac"))
    commands = []
    for algo in algorithms:
        if algo not in RL_ALGOS:
            continue
        per_algo_args = dict(eval_args)
        output_subdir = str(per_algo_args.pop("output_subdir", "eval"))
        per_algo_args["algorithms"] = algo
        per_algo_args.setdefault("group_name", exp_id)
        per_algo_args["models_root"] = relpath(exp_out / "models" / algo)
        per_algo_args["output_dir"] = relpath(exp_out / output_subdir / algo)

        cmd = [python_exe, "scripts/eval_baseline_models.py"]
        append_cli_args(cmd, per_algo_args)
        commands.append(cmd)
    return commands


def build_plot_commands(
    config: Mapping[str, Any],
    *,
    python_exe: str,
) -> list[list[str]]:
    """Build plotting/report commands from a plotting config."""
    commands: list[list[str]] = []
    for plot in config.get("plots", []) or []:
        if not isinstance(plot, Mapping):
            raise ValueError("Each plot entry must be a mapping")
        script = str(plot["script"])
        args = dict(plot.get("args", {}) or {})
        args.setdefault("output_dir", relpath(resolve_path(str(config.get("output_root", "outputs"))) / "figures" / str(plot["id"])))
        cmd = [python_exe, script]
        append_cli_args(cmd, args)
        commands.append(cmd)
    return commands


def selected_experiments(config: Mapping[str, Any], requested: str) -> list[Mapping[str, Any]]:
    """Return experiments selected by id or `all`."""
    experiments = config.get("experiments", [])
    if not isinstance(experiments, Sequence):
        raise ValueError("Config field `experiments` must be a list")
    typed = [x for x in experiments if isinstance(x, Mapping)]
    if requested == "all":
        return typed
    wanted = set(as_list(requested))
    picked = [x for x in typed if str(x.get("id")) in wanted]
    missing = wanted - {str(x.get("id")) for x in picked}
    if missing:
        raise SystemExit(f"Unknown experiment id(s): {', '.join(sorted(missing))}")
    return picked


def print_command(cmd: Sequence[str]) -> None:
    """Print a shell-escaped command."""
    print(" ".join(shlex.quote(x) for x in cmd))


def run_commands(commands: Sequence[Sequence[str]], *, dry_run: bool) -> None:
    """Run commands sequentially, or print them in dry-run mode."""
    for cmd in commands:
        print_command(cmd)
        if not dry_run:
            subprocess.run(list(cmd), cwd=REPO_ROOT, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run FPQC-SAC experiments from YAML/JSON configs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML or JSON experiment config.")
    parser.add_argument("--experiment", default="all", help="Experiment id, comma list, or `all`.")
    parser.add_argument("--stage", choices=["train", "eval", "plot", "all"], default="train")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--list", action="store_true", help="List experiment ids and exit.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child commands.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)

    if args.list:
        for experiment in selected_experiments(config, "all"):
            print(experiment["id"])
        return

    commands: list[list[str]] = []
    if args.stage in {"train", "eval", "all"}:
        for experiment in selected_experiments(config, args.experiment):
            if args.stage in {"train", "all"}:
                commands.extend(build_train_commands(config, experiment, python_exe=args.python))
            if args.stage in {"eval", "all"}:
                commands.extend(build_eval_commands(config, experiment, python_exe=args.python))
    if args.stage == "plot":
        commands.extend(build_plot_commands(config, python_exe=args.python))

    if not commands:
        raise SystemExit("No commands generated for the selected config/stage.")
    run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
