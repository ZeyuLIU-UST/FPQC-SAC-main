"""Shared data, environment, and metric utilities for reproduction scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from env import MonteCarloStockEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


TECH_INDICATORS = INDICATORS
TRANSACTION_COST_PCT = 0.0001


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"Random seed fixed: {seed}")


def load_and_engineer(
    csv_path: str,
    use_vix: bool = False,
    vix_path: Optional[str] = None,
    vix_col: str = "vix",
) -> pd.DataFrame:
    """Load a market CSV and build FinRL features."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError("File not found.")

    df = pd.read_csv(csv_file)
    df = df.rename(
        columns={
            "open": "open",
            "Open": "open",
            "high": "high",
            "High": "high",
            "low": "low",
            "Low": "low",
            "close": "close",
            "Close": "close",
            "volume": "volume",
            "Volume": "volume",
            "tic": "tic",
            "TIC": "tic",
            "symbol": "tic",
            "date": "date",
            "Date": "date",
        }
    )

    required_cols = {"date", "tic", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    if "adjcp" not in df.columns:
        df["adjcp"] = df["close"]
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECH_INDICATORS,
        use_turbulence=True,
        use_vix=False,
    )
    processed = fe.preprocess_data(df)
    processed = processed.ffill().bfill()
    processed["date"] = pd.to_datetime(processed["date"])
    processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)

    if use_vix:
        if vix_path is None:
            raise ValueError("vix_path is required when use_vix=True.")
        vix_file = Path(vix_path)
        if not vix_file.exists():
            raise FileNotFoundError("File not found.")

        vix_raw = pd.read_csv(vix_file)
        col_map = {c: c.strip().lower().replace(" ", "_") for c in vix_raw.columns}
        vix_raw = vix_raw.rename(columns=col_map)
        if "date" not in vix_raw.columns and "timestamp" in vix_raw.columns:
            vix_raw = vix_raw.rename(columns={"timestamp": "date"})
        if "close" not in vix_raw.columns:
            raise ValueError("VIX file must contain a close column.")
        if "date" not in vix_raw.columns:
            raise ValueError("VIX file must contain a date column.")

        vix_raw["date"] = pd.to_datetime(vix_raw["date"])
        vix_series = (
            vix_raw[["date", "close"]]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .rename(columns={"close": vix_col})
        )
        processed = processed.merge(vix_series, on="date", how="left")
        processed[vix_col] = processed[vix_col].ffill().bfill()

    processed.index = processed.date.factorize()[0]
    return processed


def split_train_test(
    df: pd.DataFrame,
    train_start: str = "2013-01-02",
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-02",
    test_end: str = "2021-08-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into train and test periods."""
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if len(train_df) == 0:
        raise ValueError("Training split is empty.")
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    train_df.index = train_df.date.factorize()[0]
    test_df.index = test_df.date.factorize()[0]
    print(
        f"Training set: {train_df['date'].min()} to {train_df['date'].max()}, "
        f"{len(train_df['date'].unique())} trading days"
    )
    print(
        f"Test set: {test_df['date'].min()} to {test_df['date'].max()}, "
        f"{len(test_df['date'].unique())} trading days"
    )
    return train_df, test_df


def build_env_kwargs(
    df: pd.DataFrame,
    lookahead_window: int = 5,
    reward_scaling: float = 1e-4,
    use_lookahead_reward: bool = True,
    enable_simple_mcts: bool = True,
    simple_mcts_reward_delay_steps: int = 0,
    cash_scale_factor: float = 1.0,
    observation_history_window: int = 1,
    use_vix: bool = False,
    vix_threshold: Optional[float] = None,
    vix_col: str = "vix",
) -> Dict:
    """Build keyword arguments for the trading environment."""
    stock_dim = len(df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(TECH_INDICATORS) * stock_dim

    if use_vix:
        risk_indicator_col = vix_col
        turbulence_threshold = vix_threshold
    else:
        risk_indicator_col = "turbulence"
        turbulence_threshold = None

    return {
        "hmax": 100,
        "initial_amount": 100_000,
        "stock_dim": stock_dim,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": TECH_INDICATORS,
        "reward_scaling": reward_scaling,
        "buy_cost_pct": [TRANSACTION_COST_PCT] * stock_dim,
        "sell_cost_pct": [TRANSACTION_COST_PCT] * stock_dim,
        "num_stock_shares": [0] * stock_dim,
        "turbulence_threshold": turbulence_threshold,
        "risk_indicator_col": risk_indicator_col,
        "make_plots": False,
        "print_verbosity": 10,
        "lookahead_window": lookahead_window,
        "use_lookahead_reward": use_lookahead_reward,
        "enable_simple_mcts": enable_simple_mcts,
        "simple_mcts_reward_delay_steps": simple_mcts_reward_delay_steps,
        "cash_scale_factor": float(cash_scale_factor),
        "observation_history_window": max(1, int(observation_history_window)),
    }


def make_env(df: pd.DataFrame, env_kwargs: Dict) -> MonteCarloStockEnv:
    """Create a MonteCarloStockEnv instance."""
    return MonteCarloStockEnv(df=df, **env_kwargs)


def compute_metrics(account_df: pd.DataFrame, initial_amount: float = 100_000) -> Dict[str, float]:
    """Compute return, risk, and drawdown metrics."""
    if len(account_df) == 0:
        return {}

    account_df = account_df.copy()
    total_return = (account_df["account_value"].iloc[-1] / initial_amount) - 1.0
    account_df = account_df.sort_values("date")
    account_df["daily_return"] = account_df["account_value"].pct_change().fillna(0)

    trading_days = len(account_df["daily_return"].dropna()) - 1
    annual_return = ((1 + total_return) ** (252 / trading_days)) - 1.0 if trading_days > 0 else 0.0
    annual_vol = account_df["daily_return"].std() * np.sqrt(252)
    sharpe_ratio = annual_return / (annual_vol + 1e-8)

    cummax = account_df["account_value"].cummax()
    drawdown = (account_df["account_value"] - cummax) / cummax
    max_drawdown = drawdown.min()

    downside_returns = account_df["daily_return"][account_df["daily_return"] < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        annual_downside_vol = downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        sortino_ratio = annual_return / (annual_downside_vol + 1e-8)
    else:
        sortino_ratio = sharpe_ratio

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
    }
