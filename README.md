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

<table align="center">
  <tr>
    <td align="center" width="220">
      <b>Quantum Representation</b><br/>
      <sub>Parametrized quantum circuit feature bottleneck</sub>
    </td>
    <td align="center" width="220">
      <b>Noise Suppression</b><br/>
      <sub>Stabilizes learning under low-SNR finance signals</sub>
    </td>
    <td align="center" width="220">
      <b>Stable Value Estimation</b><br/>
      <sub>Reduced critic variance vs. classical SAC</sub>
    </td>
    <td align="center" width="220">
      <b>Better Returns</b><br/>
      <sub>Smoother equity curves on held-out tests</sub>
    </td>
  </tr>
</table>

<br/>

> **SAC** — volatile critic targets and jagged test curves.  
> **FPQC-SAC** — quantum bottleneck smooths representations for steadier policy improvement.

<p align="center">
  <img src="figure/zhuyetu.png" alt="FPQC-SAC — Mitigating Bias in Low-SNR Financial RL via Quantum Representations" width="92%"/>
</p>

An anonymized research codebase for reproducing FPQC-SAC experiments: a hybrid
quantum-classical Soft Actor-Critic agent with a parametrized quantum circuit
feature bottleneck. The repository keeps the original FinRL-compatible training
stack while adding a small configuration layer for reproducible, double-blind
open-source use.

<details>
<summary><b>Table of contents</b></summary>

- [Recommended Hardware](#recommended-hardware)
- [Quick Start](#quick-start)
- [Reproduction Tutorial](#reproduction-tutorial)
- [Advanced Utilities](#advanced-utilities)
- [Output Layout](#output-layout)
- [Data](#data)
- [Reproducibility Notes](#reproducibility-notes)
- [Acknowledgements](#acknowledgements)
- [Development Checks](#development-checks)

</details>

---

## Recommended Hardware

The paper reports experiments on **Intel Core i9-13980HX + NVIDIA GeForce RTX
4090 Laptop GPU**. Use this stack when reproducing the paper's experimental
setup and baseline rankings.

| Profile | Configuration |
| --- | --- |
| **Reference (paper)** | Intel Core i9-13980HX (13th Gen, Raptor Lake) + NVIDIA GeForce RTX 4090 Laptop GPU |
| **Apple Silicon laptop** | MacBook Pro 13-inch (2020), Model MJ123ZP/A (`MacBookPro17,1`) + Apple M1 |
| **Multi-GPU server** | Intel Xeon Platinum 8581C + 4× NVIDIA L40 (48 GB) |

**Notes**

- On the RTX 4090 Laptop, keep PyTorch and PennyLane on the same CUDA device
  class when GPU quantum backends are enabled. This is the stack cited in the
  manuscript for all three portfolio CSVs
  (`repro_mainstream_tech_market_index_portfolio_2013_2023.csv`,
  `repro_defensive_blue_chip_portfolio_2013_2023.csv`,
  `repro_high_volatility_growth_portfolio_2013_2023.csv`). Independent reruns
  on the same GPU can still shift trimmed-mean CR% by noticeable margins while
  FPQC-SAC remains ahead of the SAC-family baselines.
- On Apple Silicon, PennyLane falls back to CPU. During macOS release QA we
  smoke-tested the defensive portfolio
  (`repro_defensive_blue_chip_portfolio_2013_2023.csv`) on `MacBookPro17,1` +
  M1; treat macOS as a workflow check, not the reference path for table-level
  reproduction.
- On the 4× L40 node, shard the 20-seed
  [`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt) protocol on CUDA
  (one seed block per GPU) when replaying the full three-portfolio suite at
  scale.
- The fixed CUDA scripts stop when CUDA is unavailable. For exploratory runs on
  other NVIDIA GPUs, set `ALLOW_OTHER_GPU=1`, but those runs are outside the
  reported hardware setting.

## Quick Start

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

If the four CSV files listed in the reproduction tutorial already exist at the
fixed paths, skip the download command and run the CUDA shell scripts directly.
No YAML config editing is required for the fixed reproduction workflow.

For a non-executing command preview:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_suite_cuda.sh
```

The fixed data groups are:

- `mainstream_tech_market_index`: mainstream tech and market-index portfolio: AAPL, AMZN, GOOGL, MSFT, QQQ, SPY.
- `defensive_blue_chip`: Defensive Blue-chip portfolio with traditional economy and value stocks: BAC, JNJ, JPM, PG, WMT, XOM.
- `high_volatility_growth`: High-Volatility Growth portfolio with aggressive thematic assets: AVGO, BRK.B, META, NFLX, NVDA, TSLA.

The CSV row order is part of the reproducibility protocol: for each date, rows
are written in exactly the ticker order listed above.

## Reproduction Tutorial

The recommended workflow is fixed end-to-end: download the three predefined
Yahoo panels, download the VIXY proxy, then run the CUDA scripts with
the published 20-seed list in `configs/seeds_repro_20.txt`.

### 1. Prepare Data And Seeds

Run:

```bash
bash scripts/download_reproducibility_data.sh
```

This creates:

```text
data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv
data/raw/vix_panel.csv
```

The fixed seed file is `configs/seeds_repro_20.txt`. When trim-2 is used, seeds
are sorted by test `total_return` within each model group, the two lowest and
two highest seeds are removed, and summary metrics are computed on the
remaining seeds.

### 2. Run The Fixed CUDA Core Suite

Use an NVIDIA RTX 4090 or L40 with CUDA visible to PyTorch:

```bash
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_suite_cuda.sh
```

The suite runs each group sequentially. Within a group, the algorithm order is
fixed as FPQC-SAC, PPO, DDPG, SAC, TD3, A2C, and TQC. Use `ONLY_GROUP` or
`ONLY_ALGO` only for restarting an interrupted run:

```bash
ONLY_GROUP=defensive_blue_chip ONLY_ALGO=sac PHASE=train bash scripts/run_reproducibility_suite_cuda.sh
```

For preview only:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_suite_cuda.sh
```

### 3. Run The Extended Ablation Suite

The extended suite reproduces the ablation, bottleneck, encoder, DRL baseline,
and rule-based checks. It also runs each fixed data group in order and does not
require editing YAML configs:

```bash
CUDA_VISIBLE_DEVICES=0 \
PARALLEL_WORKERS=5 \
PHASE=all \
bash scripts/run_reproducibility_extended_cuda.sh
```

Use `ONLY_GROUP` only for restarting one data group:

```bash
ONLY_GROUP=high_volatility_growth PHASE=train bash scripts/run_reproducibility_extended_cuda.sh
```

For preview only:

```bash
DRY_RUN=1 bash scripts/run_reproducibility_extended_cuda.sh
```

### 4. Generate Paper Tables

After train/eval finishes, generate trim-2 tables. If the extended suite has
been run, this command automatically finds the extended-suite `manifest.csv`
files and writes one performance table per fixed group:

```bash
bash scripts/build_analysis_tables.sh
```

For example, extended-suite tables are written under:

```text
outputs/analysis_tables/<run_prefix>/performance/
```

Each table directory contains `trim2_performance_table.tex`,
`trim2_summary_by_model.csv`, and `trim2_retained_runs.csv`. If no
extended-suite manifests are found, the script falls back to the config-based
`run.py` output layout. To select specific runs explicitly:

```bash
MANIFESTS="outputs/repro_mainstream_tech_market_index_extended_25k_rewardcache_cuda_extended/manifest.csv" \
  bash scripts/build_analysis_tables.sh
```

The performance tables report `CR%`, `AR%`, `SR`, `Sortino`, and `Calmar`.
Trim-2 means that, within each model group, seeds are sorted by test
`total_return`, the two lowest and two highest seeds are removed, and the
remaining seeds are summarized.

### 5. Generate Paper Figures

The release keeps only the figure scripts used by the paper. To evaluate the
OOS baseline curves through the end of 2023 and then plot the 2019-2023 median
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

## Advanced Utilities

The fixed reproduction workflow does not require editing YAML configs. The
`configs/` files and the `run.py` wrapper are kept for development checks,
ablation debugging, and exploratory runs outside the fixed path. For the
reported workflow, use `scripts/download_reproducibility_data.sh` and the fixed
CUDA shell scripts:

```text
scripts/run_reproducibility_suite_cuda.sh
scripts/run_reproducibility_extended_cuda.sh
```

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
`calmar_ratio`.

## Data

Full market datasets are not committed. Rebuild the fixed data panels with:

```bash
bash scripts/download_reproducibility_data.sh
```

This writes the three paper reproduction panels under `data/processed/` and the
VIXY proxy under `data/raw/vix_panel.csv`:

```text
data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv
data/raw/vix_panel.csv
```

See `data/README.md` for the schema, Yahoo download command, price adjustment
mode, and post-download validation checks.

## Reproducibility Notes

- `configs/seeds_repro_20.txt` is the fixed 20-seed list.
- The data pipeline uses Yahoo Finance from `2013-01-01` through
  `2023-12-31` (`--end 2024-01-01` because Yahoo treats `end` as exclusive),
  adjusted OHLC, and rectangular panels via `--drop-incomplete-dates`.
- The fixed CSV order is deterministic: dates ascending, and within each date
  rows follow the ticker order listed in the Quick Start.
- The reported training runs require CUDA on an NVIDIA RTX 4090 or L40.
- The main training entry point is `main_baselines_randomseed.py`.
- The unified evaluator is `scripts/eval_baseline_models.py`.
- The FPQC bottleneck implementation is in `models.py`.
- The `simple_mcts` script option is an internal delayed reward adjustment used
  by the fixed workflow; it does not require a separate planner file.

**Hardware and floating-point variance.** Even with the same seeds, data split, and
hyperparameters, numerical results can differ across machines—and sometimes
across reruns on the same machine—because training depends on device-specific
math libraries, PennyLane backend choice, and non-associative floating-point
reductions. **Use the RTX 4090 Laptop CUDA stack when checking whether FPQC-SAC
retains the top ranking** reported in the trading tables; absolute CR/AR/SR
digits are indicative, not a byte-for-byte reproduction target. High-volatility
portfolios and non-CUDA quantum simulation are the most drift-prone settings.
Treat [`configs/seeds_repro_20.txt`](configs/seeds_repro_20.txt) as the intended
experimental protocol and the trim-2 aggregator as part of the evaluation
definition; expect qualitative conclusions (SOTA ranking, risk-adjusted margins
vs. SAC-family baselines) to transfer more reliably than exact table cells.

## Acknowledgements

This project is built on a FinRL-compatible training stack. We thank the FinRL
authors and contributors for releasing their financial reinforcement learning
framework as open-source software, which made this research codebase possible.

## Development Checks

```bash
python run.py --config configs/main_fpqc_sac.yaml --stage train --dry-run
python -m py_compile run.py fpqc_sac/experiment_runner.py models.py
```

The full training run requires the processed market dataset and may take a long
time depending on seed count, timesteps, and hardware.
