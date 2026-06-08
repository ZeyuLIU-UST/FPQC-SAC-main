#!/usr/bin/env python3
"""Build trim-2 training diagnostic tables from SAC-family logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_ROW_RE = re.compile(r"\|\s*([a-zA-Z0-9_]+)\s*\|\s*([^\|]+)\|")
SEED_RE = re.compile(r"_s(\d+)")
METRICS = [
    ("q_overestimation_gap", "Q-gap$\\downarrow$"),
    ("td_target_var", "TD-var$\\downarrow$"),
    ("critic_loss_var", "C-loss-var$\\downarrow$"),
]


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML or JSON config."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("Install PyYAML to read YAML configs.") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(raw: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def as_list(value: Any) -> list[str]:
    """Normalize comma-separated strings or lists."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def experiment_records(config_paths: list[Path], eval_subdir_override: str | None) -> list[dict[str, Any]]:
    """Collect log/eval locations from experiment configs."""
    records: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        output_root = resolve_path(config.get("output_root", "outputs"))
        defaults = config.get("defaults", {})
        default_eval = defaults.get("eval", {}) if isinstance(defaults, dict) else {}
        for index, exp in enumerate(config.get("experiments", []) or []):
            if not isinstance(exp, dict):
                continue
            exp_id = str(exp["id"])
            train_cfg = exp.get("train", {}) or {}
            eval_cfg = {**default_eval, **(exp.get("eval", {}) or {})}
            algos = as_list(eval_cfg.get("algorithms") or train_cfg.get("algorithms", "sac"))
            if not algos:
                continue
            algo = algos[0]
            report = exp.get("report", {}) or {}
            eval_subdir = eval_subdir_override or str(eval_cfg.get("output_subdir", "eval"))
            records.append(
                {
                    "model_key": exp_id,
                    "label": report.get("label", eval_cfg.get("group_name", exp_id)),
                    "category": report.get("category", "Model"),
                    "order": float(report.get("order", index)),
                    "algorithm": algo,
                    "eval_dir": output_root / exp_id / eval_subdir / algo,
                    "log_dir": output_root / exp_id / "logs" / algo,
                    "config": str(config_path),
                }
            )
    return records


def to_float(raw: str) -> float | None:
    """Parse a float from a logger cell."""
    try:
        return float(raw.strip())
    except ValueError:
        return None


def extract_seed(path: Path) -> int | None:
    """Extract seed from a training log file name."""
    match = SEED_RE.search(path.name)
    return int(match.group(1)) if match else None


def parse_log(path: Path, rec: dict[str, Any]) -> pd.DataFrame:
    """Parse one SB3-style training log."""
    rows: list[dict[str, Any]] = []
    current: dict[str, float] = {}
    seed = extract_seed(path)

    def flush() -> None:
        nonlocal current
        if "total_timesteps" in current and any(metric in current for metric, _ in METRICS):
            rows.append(
                {
                    "model_key": rec["model_key"],
                    "label": rec["label"],
                    "category": rec["category"],
                    "algorithm": rec["algorithm"],
                    "seed": seed,
                    "log_file": path.name,
                    **current,
                }
            )
        current = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("---"):
            flush()
            continue
        match = LOG_ROW_RE.search(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = to_float(match.group(2))
        if value is None:
            continue
        if key == "total_timesteps":
            flush()
            current = {"total_timesteps": value}
        elif key in {metric for metric, _ in METRICS} and "total_timesteps" in current:
            current[key] = value
    flush()
    return pd.DataFrame(rows)


def trim2_seeds(eval_dir: Path) -> set[int] | None:
    """Return seeds retained by trim-2 total return; None means no eval CSV."""
    metrics_path = eval_dir / "baseline_all_model_metrics.csv"
    if not metrics_path.is_file():
        return None
    df = pd.read_csv(metrics_path)
    if "seed" not in df.columns or "total_return" not in df.columns:
        return None
    ranked = df.sort_values(["total_return", "seed"], ascending=[True, True]).reset_index(drop=True)
    kept = ranked.iloc[2:-2] if len(ranked) > 4 else ranked
    return set(int(seed) for seed in kept["seed"].dropna())


def collect(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect raw diagnostics and per-seed means, filtered by trim-2 seeds."""
    raw_frames = []
    seed_rows = []
    for rec in sorted(records, key=lambda r: (r["order"], r["model_key"])):
        log_dir = Path(rec["log_dir"])
        if not log_dir.is_dir():
            print(f"[WARN] missing log dir: {log_dir}")
            continue
        retained = trim2_seeds(Path(rec["eval_dir"]))
        for log_path in sorted(log_dir.glob("*.log")):
            log_df = parse_log(log_path, rec)
            if log_df.empty:
                continue
            if retained is not None:
                log_df = log_df[log_df["seed"].isin(retained)].copy()
            if log_df.empty:
                continue
            raw_frames.append(log_df)
            row: dict[str, Any] = {
                "model_key": rec["model_key"],
                "label": rec["label"],
                "category": rec["category"],
                "order": rec["order"],
                "algorithm": rec["algorithm"],
                "seed": int(log_df["seed"].iloc[0]) if pd.notna(log_df["seed"].iloc[0]) else np.nan,
                "log_file": log_path.name,
            }
            for metric, _ in METRICS:
                vals = pd.to_numeric(log_df.get(metric), errors="coerce").dropna()
                row[metric] = float(vals.mean()) if len(vals) else np.nan
            seed_rows.append(row)
    if not seed_rows:
        raise FileNotFoundError("No diagnostic logs were parsed. Check log directories and training output.")
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    per_seed = pd.DataFrame(seed_rows)
    return raw, per_seed


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostics over trim-2 retained seeds."""
    rows = []
    for (model_key, label, category, order), sub in per_seed.groupby(
        ["model_key", "label", "category", "order"], sort=False
    ):
        row: dict[str, Any] = {
            "model_key": model_key,
            "label": label,
            "category": category,
            "order": order,
            "n_trim2_logs": len(sub),
        }
        for metric, _ in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["order", "model_key"]).reset_index(drop=True)


def fmt_num(value: float) -> str:
    """Format a diagnostic value."""
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_tex(summary: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    """Write diagnostic LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Category & Model & " + " & ".join(name for _, name in METRICS) + r" \\",
        r"    \midrule",
    ]
    for _, row in summary.iterrows():
        cells = []
        for metric, _ in METRICS:
            cells.append(
                f"{fmt_num(float(row[f'{metric}_mean']))}$\\pm${fmt_num(float(row[f'{metric}_std']))}"
            )
        lines.append(f"    {row['category']} & {row['label']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build trim-2 Q-gap/TD-var/C-loss-var diagnostic tables.")
    parser.add_argument("--config", action="append", type=Path, required=True, help="YAML/JSON config; can repeat.")
    parser.add_argument("--eval-subdir", default=None, help="Override eval subdir used for trim-2 seed selection.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption", default="Trim-2 training diagnostics.")
    parser.add_argument("--label", default="tab:trim2_training_diagnostics")
    args = parser.parse_args()

    records = experiment_records([resolve_path(path) for path in args.config], args.eval_subdir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, per_seed = collect(records)
    summary = summarize(per_seed)
    raw.to_csv(out_dir / "diagnostic_raw_points_trim2.csv", index=False)
    per_seed.to_csv(out_dir / "diagnostic_per_seed_trim2.csv", index=False)
    summary.to_csv(out_dir / "diagnostic_summary_trim2.csv", index=False)
    write_tex(summary, out_dir / "diagnostic_qgap_tdvar_clossvar_table.tex", args.caption, args.label)
    print(out_dir / "diagnostic_qgap_tdvar_clossvar_table.tex")


if __name__ == "__main__":
    main()
