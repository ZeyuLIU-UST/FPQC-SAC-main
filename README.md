<h1 align="center">FPQC-SAC</h1>

<p align="center">
  <strong>Mitigating Bias in Low-SNR Financial Reinforcement Learning via Quantum Representations</strong>
</p>

<p align="center">
  <em>Less Noise, More Signal!</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/SAC-Soft%20Actor--Critic-5B8DEF" alt="SAC"/>
  <img src="https://img.shields.io/badge/Built--on-finrl-2EA44F" alt="Built on vendored finrl/"/>
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License MIT"/>
</p>

<p align="center">
  <img src="figure/zhuyetu.png" alt="FPQC-SAC — Mitigating Bias in Low-SNR Financial RL via Quantum Representations" width="92%"/>
</p>

A codebase for reproducing FPQC-SAC experiments: a hybrid quantum-classical Soft
Actor-Critic agent with a parametrized quantum circuit feature bottleneck,
built on a FinRL-compatible training stack.

<details>
<summary><b>Table of contents</b></summary>

- [Recommended Hardware](#recommended-hardware)
- [Data Download & Quick Start](#data-download--quick-start)
- [Fixed CUDA Scripts](#fixed-cuda-scripts)
- [Reproduction Tutorial](#reproduction-tutorial)
  - [1. Prepare Data And Seeds](#1-prepare-data-and-seeds)
  - [2. Run The Fixed CUDA Core Suite](#2-run-the-fixed-cuda-core-suite)
  - [3. Run The Extended Ablation Suite](#3-run-the-extended-ablation-suite)
  - [4. Generate Paper Tables](#4-generate-paper-tables)
  - [5. Generate Paper Figures](#5-generate-paper-figures)
- [Experiment Matrix](#experiment-matrix)
- [Advanced Utilities](#advanced-utilities)
- [Output Layout](#output-layout)
- [Reproducibility Notes](#reproducibility-notes)
- [Analysis Scripts](#analysis-scripts)
- [Acknowledgements](#acknowledgements)
- [Development Checks](#development-checks)

</details>

---

## Recommended Hardware

The paper reports experiments on **Intel Core i9-13980HX + NVIDIA GeForce RTX
4090 Laptop GPU**.

| Profile | Configuration |
| --- | --- |
| **Reference (paper)** | Intel Core i9-13980HX (13th Gen, Raptor Lake) + NVIDIA GeForce RTX 4090 Laptop GPU |
| **Apple Silicon laptop** | MacBook Pro 13-inch (2020), Model MJ123ZP/A (`MacBookPro17,1`) + Apple M1 |
| **Multi-GPU server** | Intel Xeon Platinum 8581C + 4× NVIDIA L40 (48 GB) |

**Notes**

- On the RTX 4090 Laptop, keep PyTorch and PennyLane on the same CUDA device
  when GPU quantum backends are enabled. The manuscript experiments used this
  stack for all three portfolio CSVs listed in
  [Data Download & Quick Start](#data-download--quick-start).
- On Apple Silicon, PennyLane falls back to CPU. We smoke-tested the defensive
  portfolio (`repro_defensive_blue_chip_portfolio_2013_2023.csv`) on
  `MacBookPro17,1` + M1 during release QA.
- On the 4× L40 node, shard the 20-seed
  [`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt) protocol on CUDA
  (one seed block per GPU) when replaying the full three-portfolio suite at
  scale.
- The CUDA scripts expect a visible CUDA device. Set `ALLOW_OTHER_GPU=1` to run
  on other NVIDIA GPUs.

---

## Data Download & Quick Start

Full market datasets are not committed. Place local data under:

```text
data/raw/
data/processed/
data/examples/
```

Paper experiments use **three non-overlapping U.S. six-asset portfolios**
(18 distinct assets across mainstream tech, defensive blue-chip, and
high-volatility growth regimes). Training uses **2013-01-02 through
2018-12-31**; download panels through **2024-01-01** so held-out test windows
through 2023 are covered.

### Portfolio Panels

| Group ID | Role | Tickers (in order) | Output CSV |
| --- | --- | --- | --- |
| `mainstream_tech_market_index` | Mainstream tech and market-index benchmark | AAPL, AMZN, GOOGL, MSFT, QQQ, SPY | `data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv` |
| `defensive_blue_chip` | Defensive blue-chip, lower volatility | BAC, JNJ, JPM, PG, WMT, XOM | `data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv` |
| `high_volatility_growth` | High-volatility growth, elevated regime sensitivity | AVGO, BRK.B, META, NFLX, NVDA, TSLA | `data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv` |

For each date, rows are written in the ticker order shown above. The download
commands use `--auto-adjust` and `--drop-incomplete-dates`.

### One-Command Download

```bash
bash scripts/download_reproducibility_data.sh
```

This downloads all three panels plus the VIXY proxy to:

```text
data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv
data/raw/vix_panel.csv
```

### Manual Per-Panel Download

Download each panel with
[`scripts/download_yahoo_stock_panel.py`](scripts/download_yahoo_stock_panel.py)
in this order:

```bash
# Main experiment
python scripts/download_yahoo_stock_panel.py \
  --tickers AAPL,AMZN,GOOGL,MSFT,QQQ,SPY \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv

# Sub-experiment 1
python scripts/download_yahoo_stock_panel.py \
  --tickers BAC,JNJ,JPM,PG,WMT,XOM \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv

# Sub-experiment 2
python scripts/download_yahoo_stock_panel.py \
  --tickers AVGO,BRK.B,META,NFLX,NVDA,TSLA \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv
```

If Yahoo returns no rows for `BRK.B`, retry with `BRK-B`. The downloader maps
`BRK.B` to Yahoo's `BRK-B` symbol and writes `BRK.B` back to the output CSV.

### VIX Proxy Download

```bash
python scripts/download_yahoo_stock_panel.py \
  --tickers VIXY \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/raw/vix_panel.csv
```

### Download Summary

After each download, the script prints:

```text
rows, tickers, first_date, last_date, min_rows_per_ticker,
max_rows_per_ticker, duplicate_date_tic, missing_values, rectangular
```

A complete panel typically has columns `date,Open,High,Low,Close,Volume,tic`,
with `duplicate_date_tic = 0`, `missing_values = 0`, and `rectangular = True`.
Yahoo's `end` date is exclusive, so `--end 2024-01-01` requests sessions through
the end of 2023.

See [`data/README.md`](data/README.md) for additional schema notes.

### Run Experiments

Create the conda environment from [`environment.yml`](environment.yml), download
the panels above, then launch the CUDA suites:

```bash
conda env create -f environment.yml
conda activate fpqc-sac

bash scripts/download_reproducibility_data.sh

CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_suite_cuda.sh

# Optional: extended ablations, bottlenecks, encoders, and rule-based checks.
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_extended_cuda.sh
```

If the four CSV files above are already in place, skip the download step and run
the [fixed CUDA scripts](#fixed-cuda-scripts) directly.

For a non-executing command preview:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_suite_cuda.sh
```

---

## Fixed CUDA Scripts

The fixed reproduction path uses these shell entry points:

| Script | Role |
| --- | --- |
| [`scripts/download_reproducibility_data.sh`](scripts/download_reproducibility_data.sh) | Download the three portfolio panels and VIXY proxy |
| [`scripts/run_reproducibility_suite_cuda.sh`](scripts/run_reproducibility_suite_cuda.sh) | Core FPQC-SAC and DRL baseline suite |
| [`scripts/run_reproducibility_extended_cuda.sh`](scripts/run_reproducibility_extended_cuda.sh) | Extended ablations, encoders, bottlenecks, and rule-based checks |
| [`scripts/build_analysis_tables.sh`](scripts/build_analysis_tables.sh) | Trim-2 paper tables from evaluation manifests |
| [`scripts/run_oos_baseline_curves.sh`](scripts/run_oos_baseline_curves.sh) | OOS baseline curve train/eval/plot workflow |
| [`scripts/run_latent_variance_vaf1.sh`](scripts/run_latent_variance_vaf1.sh) | Latent-variance and `VAF@1` figure pipeline |

Set `STAGE=eval` or `STAGE=plot` on the OOS wrapper, or `PHASE=train` /
`PHASE=eval` on the CUDA suite scripts, to rerun only part of a workflow:

```bash
PHASE=eval bash scripts/run_reproducibility_suite_cuda.sh
DRY_RUN=1 bash scripts/run_reproducibility_extended_cuda.sh
STAGE=plot bash scripts/run_oos_baseline_curves.sh
```

`ONLY_GROUP` and `ONLY_ALGO` are available for resuming an interrupted run. See
the [reproduction tutorial](#reproduction-tutorial) for the full step order.

---

## Reproduction Tutorial

The recommended workflow is fixed end-to-end: download the three predefined
Yahoo panels, download the VIXY proxy, then run the
[fixed CUDA scripts](#fixed-cuda-scripts) with the published 20-seed list in
[`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt).

| Step | Action |
| ---: | --- |
| 1 | [Prepare data & seeds](#1-prepare-data-and-seeds) |
| 2 | [Run the fixed CUDA core suite](#2-run-the-fixed-cuda-core-suite) |
| 3 | [Run the extended ablation suite](#3-run-the-extended-ablation-suite) |
| 4 | [Generate paper tables](#4-generate-paper-tables) |
| 5 | [Generate paper figures](#5-generate-paper-figures) |

### 1. Prepare Data And Seeds

Download the three portfolio panels and VIXY proxy as described in
[Data Download & Quick Start](#data-download--quick-start). The seed file is
[`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt). When
trim-2 is used, seeds
are sorted by test `total_return` within each model group, the two lowest and
two highest seeds are removed, and summary metrics are computed on the
remaining seeds.

### 2. Run The Fixed CUDA Core Suite

Use an NVIDIA GeForce RTX 4090 Laptop GPU or NVIDIA L40 (48 GB) with CUDA
visible to PyTorch. See [Recommended Hardware](#recommended-hardware) for the
full device notes:

```bash
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_suite_cuda.sh
```

The suite runs each group sequentially. Within a group, the algorithm order is
FPQC-SAC, PPO, DDPG, SAC, TD3, A2C, and TQC. To resume one group or algorithm:

```bash
ONLY_GROUP=defensive_blue_chip ONLY_ALGO=sac PHASE=train bash scripts/run_reproducibility_suite_cuda.sh
```

For preview only:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_suite_cuda.sh
```

### 3. Run The Extended Ablation Suite

The extended suite covers the ablation, bottleneck, encoder, DRL baseline, and
rule-based checks listed in the [experiment matrix](#experiment-matrix). It runs
each data group in order:

```bash
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_extended_cuda.sh
```

To resume one data group:

```bash
ONLY_GROUP=high_volatility_growth PHASE=train bash scripts/run_reproducibility_extended_cuda.sh
```

For preview only:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_extended_cuda.sh
```

### 4. Generate Paper Tables

After train/eval finishes, generate trim-2 tables with
[`scripts/build_analysis_tables.sh`](scripts/build_analysis_tables.sh). See
[Analysis Scripts](#analysis-scripts) for manifest overrides, output layout, and
metric definitions.

### 5. Generate Paper Figures

The release keeps only the figure scripts used by the paper. Use
[`scripts/run_oos_baseline_curves.sh`](scripts/run_oos_baseline_curves.sh) and
[`scripts/run_latent_variance_vaf1.sh`](scripts/run_latent_variance_vaf1.sh)
from the [fixed CUDA scripts](#fixed-cuda-scripts) table. To evaluate the OOS
baseline curves through the end of 2023 and then plot the 2019-2023 median
account-value figure, use the OOS workflow below. If the OOS models already
exist under the config output paths, skip the `STAGE=train` command and run only
`STAGE=eval` and `STAGE=plot`.

```bash
DATA=data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv \
  STAGE=train bash scripts/run_oos_baseline_curves.sh
DATA=data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv \
  STAGE=eval bash scripts/run_oos_baseline_curves.sh
STAGE=plot bash scripts/run_oos_baseline_curves.sh
```

The plot uses `2021-08-31` as the split between the original test window and the
later OOS window. The OOS figure uses the median account curve over all 20 seeds;
trim-2 is only used for the tables. To plot from already prepared all-seed curve
summaries:

```bash
python scripts/plot_oos_median_curves.py \
  --summary-csv outputs/figures/input/oos_median_summary.csv \
  --raw-csv outputs/figures/input/oos_all_seed_raw_curves.csv \
  --include-optional \
  --split-date 2021-08-31 \
  --test-start 2019-01-02 \
  --test-end 2023-12-31 \
  --out outputs/figures/oos_median_curves.png
```

For the latent-geometry summaries, run one script to export feature vectors,
build the latent-variance and `VAF@1` CSV files, and draw the figures:

```bash
FIGURE_GROUP_SET=encoder bash scripts/run_latent_variance_vaf1.sh
FIGURE_GROUP_SET=bottleneck bash scripts/run_latent_variance_vaf1.sh
```

The default uses the `mainstream_tech_market_index` extended-suite outputs. To use another
fixed group, set `DATA` and `RUN_PREFIX`:

```bash
DATA=data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv \
RUN_PREFIX=repro_high_volatility_growth_extended_25k_rewardcache_cuda_extended \
FIGURE_GROUP_SET=encoder \
bash scripts/run_latent_variance_vaf1.sh
```

The exported vectors are the policy feature-extractor outputs used by the actor
during evaluation. They are pooled across evaluated seeds and test steps before
the variance and `VAF@1` statistics are computed. The commands write:

```text
outputs/figures/input/baseline_all_model_feature_vectors_<group_set>.csv
outputs/figures/input/variance_by_group_<group_set>.csv
outputs/figures/input/overall_feature_shape_vaf_by_group_<group_set>.csv
outputs/figures/latent_variance_vaf1_<group_set>.png
```

---

## Experiment Matrix

Paper reproduction goes through the CUDA scripts above. The YAML configs below
are for development checks, ablation debugging, and exploratory runs:

| Config | Role |
| --- | --- |
| [`configs/main_fpqc_sac.yaml`](configs/main_fpqc_sac.yaml) | FPQC-SAC with SAC, 7 qubits, 2 PQC layers, ring entanglement |
| [`configs/baseline_sac.yaml`](configs/baseline_sac.yaml) | Classic SAC with matched seeds and data split |
| [`configs/ablations.yaml`](configs/ablations.yaml) | No-CNOT, Frozen-PQC, line entanglement, and PQC depth variants |
| [`configs/encoders_bottlenecks.yaml`](configs/encoders_bottlenecks.yaml) | RFF, Wavelet, Kalman, clipped latent, tanh / LayerNorm / weight-decay / MLP bottlenecks, SAC (no bottleneck), linear and SpectralNorm bottlenecks |
| [`configs/oos_baseline_curves.yaml`](configs/oos_baseline_curves.yaml) | OOS baseline curve train/eval/plot entry point |

Optional YAML wrappers such as
[`scripts/run_main.sh`](scripts/run_main.sh),
[`scripts/run_baselines.sh`](scripts/run_baselines.sh),
[`scripts/run_ablations.sh`](scripts/run_ablations.sh), and
[`scripts/run_encoders_bottlenecks.sh`](scripts/run_encoders_bottlenecks.sh)
call `run.py` against these configs. Paper reproduction goes through the
[fixed CUDA scripts](#fixed-cuda-scripts).

---

## Advanced Utilities

The [`configs/`](configs/) files and [`run.py`](run.py) wrapper support
development checks and exploratory runs. The paper workflow uses
[`scripts/download_reproducibility_data.sh`](scripts/download_reproducibility_data.sh)
and the [fixed CUDA scripts](#fixed-cuda-scripts). See the
[experiment matrix](#experiment-matrix) for the YAML catalog.

---

## Output Layout

All new runs write to `outputs/<experiment_id>/`:

```text
outputs/<experiment_id>/
  models/<algorithm>/
  logs/<algorithm>/
  eval/<algorithm>/
```

Evaluation always produces `baseline_all_model_metrics.csv` and per-model
`test_account_values_*.csv` files. The metrics CSV includes `group_name`,
`algorithm`, `model_name`, `seed`, `m_index`, `total_return`, `annual_return`,
`annual_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, and
`calmar_ratio`. Trim-2 table outputs are written under
[`outputs/analysis_tables/`](outputs/analysis_tables/); see
[Analysis Scripts](#analysis-scripts).

---

## Reproducibility Notes

- [`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt) is the 20-seed list
  used by the CUDA scripts. [`configs/seeds.example.txt`](configs/seeds.example.txt)
  shows the one-line seed-file format.
- The data pipeline uses Yahoo Finance from `2013-01-01` through
  `2023-12-31` (`--end 2024-01-01` because Yahoo treats `end` as exclusive),
  adjusted OHLC, and rectangular panels via `--drop-incomplete-dates`.
- CSV rows are sorted by ascending date; within each date they follow the ticker
  order listed in [Data Download & Quick Start](#data-download--quick-start).
- Paper training runs used an NVIDIA GeForce RTX 4090 Laptop GPU or NVIDIA L40
  (48 GB). See [Recommended Hardware](#recommended-hardware).
- The main training entry point is
  [`main_baselines_randomseed.py`](main_baselines_randomseed.py).
- The unified evaluator is
  [`scripts/eval_baseline_models.py`](scripts/eval_baseline_models.py).
- The FPQC bottleneck implementation is in [`models.py`](models.py).
- The `simple_mcts` script option is an internal delayed reward adjustment; no
  separate planner file is involved.

**Hardware and floating-point variance.** Results can vary slightly across
machines because of device-specific numerics. Paper experiments used an
**NVIDIA GeForce RTX 4090 Laptop GPU** or **NVIDIA L40 (48 GB)**. The seed
list is [`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt); tables use
trim-2 aggregation.

---

## Analysis Scripts

After training and evaluation, run the bundled analysis pipeline:

```bash
bash scripts/build_analysis_tables.sh
```

If the extended suite has been run, this command automatically finds the
extended-suite `manifest.csv` files and writes one performance table per fixed
group. For example, extended-suite tables are written under:

```text
outputs/analysis_tables/<run_prefix>/performance/
```

Each table directory contains `trim2_performance_table.tex`,
`trim2_summary_by_model.csv`, and `trim2_retained_runs.csv`. If no
extended-suite manifests are found, the script falls back to the config-based
[`run.py`](run.py) output layout. To select specific runs explicitly:

```bash
MANIFESTS="outputs/repro_mainstream_tech_market_index_extended_25k_rewardcache_cuda_extended/manifest.csv" \
  bash scripts/build_analysis_tables.sh
```

The performance tables report `CR%`, `AR%`, `SR`, `Sortino`, and `Calmar`.
Trim-2 means that, within each model group, seeds are sorted by test
`total_return`, the two lowest and two highest seeds are removed, and the
remaining seeds are summarized.

---

## Acknowledgements

This project is built on a FinRL-compatible training stack. We thank the FinRL
authors and contributors for releasing their financial reinforcement learning
framework as open-source software, which made this research codebase possible.

We also thank Yang LI and Xiyue LIU, research assistants at HKUST, for their
help with the experiments.

---

## Development Checks

```bash
python run.py --config configs/main_fpqc_sac.yaml --stage train --dry-run
python -m py_compile run.py fpqc_sac/experiment_runner.py models.py
```

The full training run requires the processed market dataset and may take a long
time depending on seed count, timesteps, and hardware.
