#!/usr/bin/env python3
"""Exact comparison for full-suite `baseline_all_model_metrics.csv` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
]


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a full-suite manifest."""
    df = pd.read_csv(path)
    required = {"model_key", "eval_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return df


def metrics_path(root: Path, eval_dir: str) -> Path:
    """Return metrics CSV path for an eval directory."""
    return root / eval_dir / "baseline_all_model_metrics.csv"


def compare_one(model_key: str, old_csv: Path, new_csv: Path) -> list[dict[str, str]]:
    """Compare one model group's metrics exactly."""
    errors: list[dict[str, str]] = []
    if not old_csv.is_file():
        return [{"model_key": model_key, "issue": "missing_old", "detail": str(old_csv)}]
    if not new_csv.is_file():
        return [{"model_key": model_key, "issue": "missing_new", "detail": str(new_csv)}]

    old = pd.read_csv(old_csv)
    new = pd.read_csv(new_csv)
    if list(old.columns) != list(new.columns):
        errors.append(
            {
                "model_key": model_key,
                "issue": "columns_differ",
                "detail": f"old={list(old.columns)} new={list(new.columns)}",
            }
        )
        common_cols = [c for c in old.columns if c in new.columns]
        old = old[common_cols]
        new = new[common_cols]

    sort_cols = [c for c in ["seed", "m_index", "model_name", "algorithm"] if c in old.columns and c in new.columns]
    if sort_cols:
        old = old.sort_values(sort_cols).reset_index(drop=True)
        new = new.sort_values(sort_cols).reset_index(drop=True)

    if old.shape != new.shape:
        errors.append(
            {
                "model_key": model_key,
                "issue": "shape_differ",
                "detail": f"old={old.shape} new={new.shape}",
            }
        )
        return errors

    compare_cols = [c for c in METRIC_COLUMNS if c in old.columns and c in new.columns]
    id_cols = [c for c in ["seed", "m_index", "model_name", "algorithm", "group_name"] if c in old.columns and c in new.columns]
    for col in id_cols + compare_cols:
        old_s = old[col]
        new_s = new[col]
        neq = ~(old_s.fillna("<NA>").astype(str) == new_s.fillna("<NA>").astype(str))
        if neq.any():
            idx = int(neq[neq].index[0])
            errors.append(
                {
                    "model_key": model_key,
                    "issue": f"value_differ:{col}",
                    "detail": f"row={idx} old={old_s.iloc[idx]!r} new={new_s.iloc[idx]!r}",
                }
            )
    return errors


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare full-suite eval metrics exactly.")
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    old_manifest = read_manifest(args.old_manifest).set_index("model_key")
    new_manifest = read_manifest(args.new_manifest).set_index("model_key")
    model_keys = sorted(set(old_manifest.index) & set(new_manifest.index))
    all_errors: list[dict[str, str]] = []
    for key in model_keys:
        old_csv = metrics_path(args.old_root, str(old_manifest.loc[key, "eval_dir"]))
        new_csv = metrics_path(args.new_root, str(new_manifest.loc[key, "eval_dir"]))
        all_errors.extend(compare_one(key, old_csv, new_csv))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(all_errors, columns=["model_key", "issue", "detail"])
    out.to_csv(args.output_csv, index=False)
    if all_errors:
        print(f"Found {len(all_errors)} differences -> {args.output_csv}")
        raise SystemExit(1)
    print(f"Exact match for {len(model_keys)} model groups.")


if __name__ == "__main__":
    main()
