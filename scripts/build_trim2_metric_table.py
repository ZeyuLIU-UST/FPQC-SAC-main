#!/usr/bin/env python3
"""Build trim-2 performance tables from standardized evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("total_return", "CR\\%$\\uparrow$", True),
    ("annual_return", "AR\\%$\\uparrow$", True),
    ("sharpe_ratio", "SR$\\uparrow$", False),
    ("sortino_ratio", "Sortino$\\uparrow$", False),
    ("calmar_ratio", "Calmar$\\uparrow$", False),
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
    """Normalize a scalar or list value to strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def experiment_records(config_paths: list[Path], eval_subdir_override: str | None) -> list[dict[str, Any]]:
    """Collect report records from experiment configs."""
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
            eval_cfg = {**default_eval, **(exp.get("eval", {}) or {})}
            algos = as_list(eval_cfg.get("algorithms") or (exp.get("train", {}) or {}).get("algorithms", "sac"))
            if not algos:
                continue
            report = exp.get("report", {}) or {}
            eval_subdir = eval_subdir_override or str(eval_cfg.get("output_subdir", "eval"))
            records.append(
                {
                    "model_key": exp_id,
                    "label": report.get("label", eval_cfg.get("group_name", exp_id)),
                    "category": report.get("category", "Model"),
                    "order": float(report.get("order", index)),
                    "algorithm": algos[0],
                    "eval_dir": output_root / exp_id / eval_subdir / algos[0],
                    "config": str(config_path),
                }
            )
    return records


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load records from a CSV manifest."""
    df = pd.read_csv(path)
    required = {"model_key", "label", "eval_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    records = []
    for idx, row in df.reset_index(drop=True).iterrows():
        records.append(
            {
                "model_key": row["model_key"],
                "label": row["label"],
                "category": row.get("category", "Model"),
                "order": float(row.get("order", idx)),
                "algorithm": row.get("eval_algo", row.get("algorithm", "")),
                "eval_dir": resolve_path(row["eval_dir"]),
                "config": str(path),
            }
        )
    return records


def trim2(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the two lowest and two highest total-return seeds when possible."""
    ranked = df.sort_values(["total_return", "seed"], ascending=[True, True]).reset_index(drop=True)
    if len(ranked) <= 4:
        return ranked.copy()
    return ranked.iloc[2:-2].copy()


def fmt_num(value: float) -> str:
    """Format table cells compactly."""
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.2f}"


def fmt_pm(mean: float, std: float) -> str:
    """Format mean +/- std for LaTeX."""
    return f"{fmt_num(mean)}$\\pm${fmt_num(std)}"


def summarize(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics, retain trim-2 rows, and summarize each model."""
    retained_frames = []
    summary_rows = []
    for rec in sorted(records, key=lambda r: (r["order"], r["model_key"])):
        metrics_path = Path(rec["eval_dir"]) / "baseline_all_model_metrics.csv"
        if not metrics_path.is_file():
            print(f"[WARN] missing metrics: {metrics_path}")
            continue
        df = pd.read_csv(metrics_path)
        if "total_return" not in df.columns:
            raise ValueError(f"Missing total_return column: {metrics_path}")
        kept = trim2(df)
        for key in ["model_key", "label", "category", "order", "algorithm", "eval_dir", "config"]:
            kept[key] = rec[key]
        retained_frames.append(kept)

        row: dict[str, Any] = {
            "model_key": rec["model_key"],
            "label": rec["label"],
            "category": rec["category"],
            "order": rec["order"],
            "algorithm": rec["algorithm"],
            "n_all": len(df),
            "n_trim2": len(kept),
        }
        for col, _, as_pct in METRICS:
            vals = pd.to_numeric(kept[col], errors="coerce").dropna()
            if as_pct:
                vals = vals * 100.0
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    if not summary_rows:
        raise FileNotFoundError("No metrics were loaded. Check configs/manifests and eval directories.")
    retained = pd.concat(retained_frames, ignore_index=True) if retained_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["order", "model_key"]).reset_index(drop=True)
    return retained, summary


def write_tex(summary: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    """Write a compact LaTeX table for trim-2 metrics."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        r"  \begin{tabular}{llccccc}",
        r"    \toprule",
        r"    Category & Model & " + " & ".join(name for _, name, _ in METRICS) + r" \\",
        r"    \midrule",
    ]
    for _, row in summary.iterrows():
        cells = []
        for col, _, _ in METRICS:
            cells.append(fmt_pm(float(row[f"{col}_mean"]), float(row[f"{col}_std"])))
        lines.append(f"    {row['category']} & {row['label']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build trim-2 CR/AR/SR/Sortino/Calmar CSV and TeX tables.")
    parser.add_argument("--config", action="append", type=Path, default=[], help="YAML/JSON config; can repeat.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV manifest with model_key,label,eval_dir.")
    parser.add_argument("--eval-subdir", default=None, help="Override eval subdir, e.g. eval_oos.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption", default="Trim-2 performance comparison.")
    parser.add_argument("--label", default="tab:trim2_performance")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if args.manifest is not None:
        records.extend(load_manifest(resolve_path(args.manifest)))
    if args.config:
        records.extend(experiment_records([resolve_path(p) for p in args.config], args.eval_subdir))
    if not records:
        raise SystemExit("Provide at least one --config or --manifest.")

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retained, summary = summarize(records)
    retained.to_csv(out_dir / "trim2_retained_runs.csv", index=False)
    summary.to_csv(out_dir / "trim2_summary_by_model.csv", index=False)
    write_tex(summary, out_dir / "trim2_performance_table.tex", args.caption, args.label)
    print(out_dir / "trim2_performance_table.tex")


if __name__ == "__main__":
    main()
