#!/usr/bin/env python3
"""Build latent-variance and VAF@1 input CSV files from exported features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_GROUP_LABELS = {
    "fpqc_sac": "FPQC-SAC",
    "sac_quantum": "FPQC-SAC",
    "rff": "RFF",
    "fourier": "RFF",
    "sac_fourier": "RFF",
    "wavelet": "Wavelet",
    "sac_wavelet": "Wavelet",
    "kalman": "Kalman",
    "sac_kalman": "Kalman",
    "clipped_latent": "Clipped Latent",
    "clipped_bottleneck": "Clipped Latent",
    "sac_clipped_bottleneck": "Clipped Latent",
    "tanh_bottleneck": "Tanh Bottleneck",
    "sac_tanh_bottleneck": "Tanh Bottleneck",
    "layernorm_bottleneck": "LayerNorm Bot.",
    "sac_layernorm_bottleneck": "LayerNorm Bot.",
    "weight_decay_bottleneck": "Weight Decay Bot.",
    "sac_weight_decay_bottleneck": "Weight Decay Bot.",
    "mlp_bottleneck": "MLP Bottleneck",
    "sac_mlp": "MLP Bottleneck",
    "sac_no_bottleneck": "SAC (No Bottleneck)",
    "sac": "SAC (No Bottleneck)",
    "linear_bottleneck": "Linear Bottleneck",
    "sac_linear_bottleneck": "Linear Bottleneck",
    "spectralnorm_bottleneck": "SpectralNorm Bot.",
    "spectral_bottleneck": "SpectralNorm Bot.",
    "sac_spectral_bottleneck": "SpectralNorm Bot.",
}


DEFAULT_GROUP_ORDER = [
    "fpqc_sac",
    "sac_quantum",
    "rff",
    "fourier",
    "sac_fourier",
    "wavelet",
    "sac_wavelet",
    "kalman",
    "sac_kalman",
    "clipped_latent",
    "clipped_bottleneck",
    "sac_clipped_bottleneck",
    "tanh_bottleneck",
    "sac_tanh_bottleneck",
    "layernorm_bottleneck",
    "sac_layernorm_bottleneck",
    "weight_decay_bottleneck",
    "sac_weight_decay_bottleneck",
    "mlp_bottleneck",
    "sac_mlp",
    "sac_no_bottleneck",
    "sac",
    "linear_bottleneck",
    "sac_linear_bottleneck",
    "spectralnorm_bottleneck",
    "spectral_bottleneck",
    "sac_spectral_bottleneck",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns named f0, f1, ... in order."""
    cols = [col for col in df.columns if col.startswith("f") and col[1:].isdigit()]
    cols = sorted(cols, key=lambda name: int(name[1:]))
    if not cols:
        raise ValueError("No feature columns found. Expected columns named f0, f1, ...")
    return cols


def vaf_percent(x: np.ndarray, n_components: int) -> float:
    """Compute PCA reconstruction VAF for one group."""
    n_components = min(n_components, x.shape[0], x.shape[1])
    if n_components <= 0:
        return float("nan")
    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(x)
    x_hat = pca.inverse_transform(z)
    residual_var = np.var(x - x_hat, ddof=1)
    total_var = np.var(x, ddof=1)
    if total_var == 0:
        return float("nan")
    return float((1.0 - residual_var / total_var) * 100.0)


def parse_groups(raw: str | None) -> list[str] | None:
    """Parse an optional comma-separated group filter."""
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def ordered_groups(df: pd.DataFrame, requested_groups: list[str] | None) -> list[str]:
    """Choose groups in a stable order."""
    available = set(df["group_name"].astype(str))
    if requested_groups is not None:
        missing = [group for group in requested_groups if group not in available]
        if missing:
            raise ValueError(f"Requested groups not found in feature CSV: {missing}")
        return requested_groups
    ordered = [group for group in DEFAULT_GROUP_ORDER if group in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def load_features(path: Path, requested_groups: list[str] | None) -> pd.DataFrame:
    """Load and validate exported feature vectors."""
    df = pd.read_csv(path)
    if "group_name" not in df.columns:
        raise ValueError(f"{path} missing required column: group_name")
    cols = feature_columns(df)
    df = df.copy()
    df["group_name"] = df["group_name"].astype(str)
    groups = ordered_groups(df, requested_groups)
    df = df[df["group_name"].isin(groups)].copy()
    if df.empty:
        raise ValueError("No feature rows remain after applying group filters.")
    sort_cols = [col for col in ["group_name", "model_name", "train_seed", "step", "date"] if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)


def compute_stats(df: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute total latent variance and VAF summaries by group."""
    cols = feature_columns(df)
    x_raw = df[cols].to_numpy(dtype=np.float64)
    x_std = StandardScaler().fit_transform(x_raw)
    group_names = df["group_name"].to_numpy()

    variance_rows: list[dict[str, object]] = []
    vaf_rows: list[dict[str, object]] = []
    for group in groups:
        mask = group_names == group
        if not np.any(mask):
            continue
        raw_group = x_raw[mask]
        std_group = x_std[mask]
        raw_center = raw_group.mean(axis=0)
        std_center = std_group.mean(axis=0)
        raw_dists = np.linalg.norm(raw_group - raw_center, axis=1)
        std_dists = np.linalg.norm(std_group - std_center, axis=1)
        label = DEFAULT_GROUP_LABELS.get(group, group)
        variance_rows.append(
            {
                "group_name": group,
                "label": label,
                "n_vectors": int(mask.sum()),
                "raw_total_variance": float(np.var(raw_group, axis=0, ddof=1).sum()),
                "raw_mean_radius": float(raw_dists.mean()),
                "std_total_variance": float(np.var(std_group, axis=0, ddof=1).sum()),
                "std_mean_radius": float(std_dists.mean()),
                "std_median_radius": float(np.median(std_dists)),
            }
        )
        vaf_rows.append(
            {
                "group_name": group,
                "label": label,
                "n_vectors": int(mask.sum()),
                "vaf_1_pct": vaf_percent(std_group, 1),
                "vaf_3_pct": vaf_percent(std_group, 3),
                "vaf_5_pct": vaf_percent(std_group, 5),
                "vaf_10_pct": vaf_percent(std_group, 10),
            }
        )
    return pd.DataFrame(variance_rows), pd.DataFrame(vaf_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latent variance and VAF@1 CSV inputs.")
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("outputs/figures/input/baseline_all_model_feature_vectors.csv"),
        help="Feature vector CSV exported by scripts/eval_baseline_models.py --export-feature-vectors.",
    )
    parser.add_argument(
        "--variance-csv",
        type=Path,
        default=Path("outputs/figures/input/variance_by_group.csv"),
    )
    parser.add_argument(
        "--vaf-csv",
        type=Path,
        default=Path("outputs/figures/input/overall_feature_shape_vaf_by_group.csv"),
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Optional comma-separated group_name filter. Defaults to all known disclosed groups found in the CSV.",
    )
    args = parser.parse_args()

    requested_groups = parse_groups(args.groups)
    features = load_features(args.feature_csv, requested_groups)
    groups = ordered_groups(features, requested_groups)
    variance_df, vaf_df = compute_stats(features, groups)

    args.variance_csv.parent.mkdir(parents=True, exist_ok=True)
    args.vaf_csv.parent.mkdir(parents=True, exist_ok=True)
    variance_df.to_csv(args.variance_csv, index=False)
    vaf_df.to_csv(args.vaf_csv, index=False)
    print(args.variance_csv)
    print(args.vaf_csv)


if __name__ == "__main__":
    main()
