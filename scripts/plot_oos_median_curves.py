#!/usr/bin/env python3
"""Plot OOS median account-value curves from all-seed curve CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "FPQC-SAC": "#d62728",
    "SAC": "#17becf",
    "TD3": "#ff7f0e",
    "PPO": "#1f77b4",
    "DDPG": "#9467bd",
    "A2C": "#2ca02c",
    "TQC": "#8c6d31",
    "SPY (B&H)": "#000000",
    "Equal Weight (B&H)": "#6b6b6b",
    "MACD": "#8c564b",
    "SMA": "#e377c2",
    "KDJ_RSI": "#bcbd22",
    "ZMR": "#7f7f7f",
}

DEFAULT_ORDER = ["FPQC-SAC", "SAC", "TD3", "PPO", "DDPG", "A2C", "TQC"]
RAW_OPTIONAL_COLUMNS = {
    "SPY_(B&H)": "SPY (B&H)",
    "Equal_Weight_(B&H)": "Equal Weight (B&H)",
    "Rule_MACD": "MACD",
    "Rule_SMA": "SMA",
    "Rule_KDJ_RSI": "KDJ_RSI",
    "Rule_ZMR": "ZMR",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SystemExit("YAML configs require PyYAML.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def _config_records(config_paths: list[Path], eval_subdir_override: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = _load_config(config_path)
        output_root = _resolve_path(str(config.get("output_root", "outputs")))
        defaults = config.get("defaults", {})
        default_eval = defaults.get("eval", {}) if isinstance(defaults, dict) else {}
        for index, exp in enumerate(config.get("experiments", []) or []):
            if not isinstance(exp, dict):
                continue
            exp_id = str(exp["id"])
            train_cfg = exp.get("train", {}) or {}
            eval_cfg = {**default_eval, **(exp.get("eval", {}) or {})}
            algos = _as_list(eval_cfg.get("algorithms") or train_cfg.get("algorithms", "sac"))
            if not algos:
                continue
            report = exp.get("report", {}) or {}
            eval_subdir = eval_subdir_override or str(eval_cfg.get("output_subdir", "eval"))
            records.append(
                {
                    "model_key": exp_id,
                    "label": str(report.get("label", eval_cfg.get("group_name", exp_id))),
                    "order": float(report.get("order", index)),
                    "algorithm": algos[0],
                    "eval_dir": output_root / exp_id / eval_subdir / algos[0],
                }
            )
    return records


def _read_account_curve(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "date" not in df.columns or "account_value" not in df.columns:
        raise ValueError(f"{path} missing date/account_value columns")
    return pd.Series(
        pd.to_numeric(df["account_value"], errors="coerce").values,
        index=pd.to_datetime(df["date"]),
        name=path.stem,
    ).dropna().sort_index()


def _curves_from_eval_records(records: list[dict[str, Any]]) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    summary_series: dict[str, pd.Series] = {}
    raw_frames: list[pd.DataFrame] = []
    for rec in sorted(records, key=lambda item: (item["order"], item["label"])):
        eval_dir = Path(rec["eval_dir"])
        metrics_path = eval_dir / "baseline_all_model_metrics.csv"
        if not metrics_path.is_file():
            print(f"[WARN] missing metrics: {metrics_path}")
            continue
        metrics = pd.read_csv(metrics_path)
        if "total_return" not in metrics.columns or "model_name" not in metrics.columns:
            raise ValueError(f"{metrics_path} missing total_return/model_name columns")
        all_seed_rows = metrics.sort_values(["seed", "model_name"], ascending=[True, True]).reset_index(drop=True)
        curves = []
        for _, row in all_seed_rows.iterrows():
            model_name = str(row["model_name"])
            curve_path = eval_dir / f"test_account_values_{model_name}.csv"
            if not curve_path.is_file():
                print(f"[WARN] missing curve: {curve_path}")
                continue
            curve = _read_account_curve(curve_path)
            curve.name = f"{rec['label']}_s{int(row['seed'])}"
            curves.append(curve)
        if not curves:
            continue
        wide = pd.concat(curves, axis=1).sort_index()
        label = str(rec["label"])
        summary_series[label] = wide.median(axis=1).dropna()
        raw_frames.append(wide)
    if not summary_series:
        raise FileNotFoundError("No OOS curves were loaded from the selected config.")
    raw_df = pd.concat(raw_frames, axis=1).sort_index() if raw_frames else pd.DataFrame()
    return summary_series, raw_df


def _write_summary_csv(summary_series: dict[str, pd.Series], out_csv: Path) -> None:
    rows = []
    for label, series in summary_series.items():
        rows.append(series.rename(f"{label}_median"))
    summary_df = pd.concat(rows, axis=1).sort_index()
    summary_df.index.name = "date"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv)


def _read_summary(summary_csv: Path, model_order: list[str]) -> dict[str, pd.Series]:
    df = pd.read_csv(summary_csv)
    if "date" not in df.columns:
        raise ValueError(f"{summary_csv} missing date column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    series: dict[str, pd.Series] = {}
    for label in model_order:
        col = f"{label}_median"
        if col in df.columns:
            series[label] = pd.to_numeric(df[col], errors="coerce").dropna()
    if not series:
        raise ValueError(f"No *_median columns found in {summary_csv}")
    return series


def _read_optional_raw(raw_csv: Path | None) -> dict[str, pd.Series]:
    if raw_csv is None:
        return {}
    df = pd.read_csv(raw_csv)
    if "date" not in df.columns:
        raise ValueError(f"{raw_csv} missing date column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out: dict[str, pd.Series] = {}
    for col, label in RAW_OPTIONAL_COLUMNS.items():
        if col in df.columns:
            out[label] = pd.to_numeric(df[col], errors="coerce").dropna()
    return out


def plot(
    summary_series: dict[str, pd.Series],
    optional_series: dict[str, pd.Series],
    out_png: Path,
    *,
    split_date: pd.Timestamp,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
    include_optional: bool,
) -> None:
    _set_style()
    all_series = dict(summary_series)
    if include_optional:
        all_series.update(optional_series)

    if test_start is None:
        test_start = min(s.index.min() for s in all_series.values())
    if test_end is None:
        test_end = max(s.index.max() for s in all_series.values())

    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    right_start = split_date + pd.Timedelta(days=1)
    ax.axvspan(test_start, split_date, color="#d9eaf7", alpha=0.35, zorder=0)
    ax.axvspan(right_start, test_end, color="#fce5cd", alpha=0.35, zorder=0)
    ax.axvline(split_date, color="#666666", linestyle="--", linewidth=1.4, zorder=1)

    for label, series in all_series.items():
        series = series[(series.index >= test_start) & (series.index <= test_end)]
        if series.empty:
            continue
        is_main = label == "FPQC-SAC"
        linestyle = "--" if label == "Equal Weight (B&H)" else (":" if label == "SPY (B&H)" else "-")
        ax.plot(
            series.index,
            series.values,
            label=label,
            color=COLORS.get(label, "#888888"),
            linewidth=3.0 if is_main else 1.8,
            alpha=1.0 if is_main else 0.9,
            linestyle=linestyle,
            zorder=5 if is_main else 3,
        )

    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.92 * (ymax - ymin)
    left_mid = test_start + (split_date - test_start) / 2
    right_mid = right_start + (test_end - right_start) / 2
    ax.text(left_mid, y_text, "Original experiment", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(right_mid, y_text, "Further OOS", ha="center", va="center", fontsize=15, fontweight="bold")

    ax.set_title("Median Account Curves with OOS Extrapolation", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, ncol=1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OOS median account-value curves.")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--config", action="append", type=Path, default=[], help="OOS config used to locate eval outputs.")
    parser.add_argument("--eval-subdir", default=None, help="Override eval subdir, for example eval_oos.")
    parser.add_argument("--raw-csv", type=Path, default=None, help="Optional raw all-seed curve CSV with SPY/rule columns.")
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/oos_median_curves.png"))
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--write-summary-csv", type=Path, default=None, help="Optional path for generated median summary CSV.")
    parser.add_argument("--write-raw-csv", type=Path, default=None, help="Optional path for generated all-seed raw curves CSV.")
    parser.add_argument("--split-date", type=str, default="2021-08-31")
    parser.add_argument("--test-start", type=str, default="")
    parser.add_argument("--test-end", type=str, default="")
    parser.add_argument("--model-order", type=str, default=",".join(DEFAULT_ORDER))
    parser.add_argument("--include-optional", action="store_true", help="Include SPY/equal-weight/rule curves from --raw-csv.")
    args = parser.parse_args()

    model_order = [x.strip() for x in args.model_order.split(",") if x.strip()]
    if args.config:
        records = _config_records([_resolve_path(path) for path in args.config], args.eval_subdir)
        summary_series, raw_df = _curves_from_eval_records(records)
        if args.write_summary_csv:
            _write_summary_csv(summary_series, _resolve_path(args.write_summary_csv))
        if args.write_raw_csv and not raw_df.empty:
            out_raw = _resolve_path(args.write_raw_csv)
            out_raw.parent.mkdir(parents=True, exist_ok=True)
            raw_df.index.name = "date"
            raw_df.to_csv(out_raw)
    else:
        if args.summary_csv is None:
            raise SystemExit("Provide --summary-csv or --config.")
        summary_series = _read_summary(_resolve_path(args.summary_csv), model_order)
    optional_series = _read_optional_raw(_resolve_path(args.raw_csv) if args.raw_csv else None)
    plot(
        summary_series,
        optional_series,
        _resolve_path(args.out),
        split_date=pd.Timestamp(args.split_date),
        test_start=pd.Timestamp(args.test_start) if args.test_start else None,
        test_end=pd.Timestamp(args.test_end) if args.test_end else None,
        include_optional=args.include_optional,
    )
    print(args.out)


if __name__ == "__main__":
    main()
