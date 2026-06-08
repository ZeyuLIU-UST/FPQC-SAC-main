#!/usr/bin/env python3
"""Plot total latent variance and VAF@1 from summary CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LABELS = {
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
    "fpqc_no_cnot": "FPQC no-CNOT",
    "sac_no_cnot": "FPQC no-CNOT",
    "fpqc_frozen": "FPQC frozen-rot",
    "sac_freeze_pqc": "FPQC frozen-rot",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def _label(row: pd.Series) -> str:
    if "label" in row and pd.notna(row["label"]):
        return str(row["label"])
    return DEFAULT_LABELS.get(str(row["group_name"]), str(row["group_name"]))


def _colors(groups: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab20")
    color_map = {group: cmap(i % 20) for i, group in enumerate(groups)}
    color_map["fpqc_sac"] = "#e41a1c"
    color_map["sac_quantum"] = "#e41a1c"
    color_map["rff"] = "#ff7f0e"
    color_map["fourier"] = "#ff7f0e"
    color_map["sac_no_cnot"] = "#9467bd"
    color_map["fpqc_no_cnot"] = "#9467bd"
    color_map["sac_fourier"] = "#ff7f0e"
    return color_map


def load_inputs(variance_csv: Path, vaf_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    variance_df = pd.read_csv(variance_csv)
    vaf_df = pd.read_csv(vaf_csv)
    for path, df, required in [
        (variance_csv, variance_df, {"group_name", "std_total_variance"}),
        (vaf_csv, vaf_df, {"group_name", "vaf_1_pct"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return variance_df, vaf_df


def plot(variance_df: pd.DataFrame, vaf_df: pd.DataFrame, out_png: Path) -> None:
    _set_style()
    groups = [g for g in variance_df["group_name"].astype(str).tolist() if g in set(vaf_df["group_name"].astype(str))]
    color_map = _colors(groups)

    variance_sorted = variance_df.copy()
    variance_sorted["group_name"] = variance_sorted["group_name"].astype(str)
    variance_sorted = variance_sorted[variance_sorted["group_name"].isin(groups)]
    variance_sorted = variance_sorted.sort_values("std_total_variance", ascending=True).reset_index(drop=True)

    vaf_sorted = vaf_df.copy()
    vaf_sorted["group_name"] = vaf_sorted["group_name"].astype(str)
    vaf_sorted = vaf_sorted[vaf_sorted["group_name"].isin(groups)]
    vaf_sorted = vaf_sorted.sort_values("vaf_1_pct", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.8), gridspec_kw={"wspace": 0.42})

    y1 = np.arange(len(variance_sorted))
    ax1.barh(
        y1,
        variance_sorted["std_total_variance"].astype(float),
        color=[color_map[g] for g in variance_sorted["group_name"]],
        alpha=0.88,
    )
    ax1.set_yticks(y1)
    ax1.set_yticklabels([_label(row) for _, row in variance_sorted.iterrows()])
    ax1.set_xscale("log")
    ax1.set_xlabel("Total latent variance (log scale)")
    ax1.set_title("Total Latent Variance")
    ax1.grid(True, axis="x", alpha=0.22)

    y2 = np.arange(len(vaf_sorted))
    ax2.barh(
        y2,
        vaf_sorted["vaf_1_pct"].astype(float),
        color=[color_map[g] for g in vaf_sorted["group_name"]],
        alpha=0.88,
    )
    ax2.set_yticks(y2)
    ax2.set_yticklabels([_label(row) for _, row in vaf_sorted.iterrows()])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("VAF@1 (%)")
    ax2.set_title("Feature-Shape Reconstruction VAF@1")
    ax2.grid(True, axis="x", alpha=0.22)
    for y, value in zip(y2, vaf_sorted["vaf_1_pct"].astype(float)):
        ax2.text(min(value + 1.0, 98.0), y, f"{value:.2f}%", va="center", fontsize=11)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.90, wspace=0.42)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot total latent variance and VAF@1.")
    parser.add_argument("--variance-csv", type=Path, required=True)
    parser.add_argument("--vaf-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/latent_variance_vaf1.png"))
    args = parser.parse_args()

    variance_df, vaf_df = load_inputs(args.variance_csv, args.vaf_csv)
    plot(variance_df, vaf_df, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
