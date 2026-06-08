"""FinRL-compatible stock trading environment with look-ahead reward support."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class MonteCarloStockEnv(StockTradingEnv):
    """StockTradingEnv variant used by the reproduction scripts."""

    def __init__(
        self,
        *args,
        lookahead_window: int = 5,
        enable_simple_mcts: bool = True,
        use_lookahead_reward: bool = True,
        simple_mcts_reward_delay_steps: int = 0,
        cash_scale_factor: float = 1.0,
        observation_history_window: int = 1,
        **kwargs,
    ):
        self.lookahead_window = lookahead_window
        self.enable_simple_mcts = enable_simple_mcts
        self.use_lookahead_reward = use_lookahead_reward
        self.simple_mcts_reward_delay_steps = max(0, int(simple_mcts_reward_delay_steps))
        self.cash_scale_factor = float(cash_scale_factor or 1.0)
        if self.cash_scale_factor <= 0:
            raise ValueError(f"cash_scale_factor must be > 0, got: {cash_scale_factor}")
        self.observation_history_window = max(1, int(observation_history_window))
        self._obs_history: list[np.ndarray] = []
        self._simple_mcts_pending_snapshots: Dict[int, List[Tuple[int, float, np.ndarray]]] = {}
        super().__init__(*args, **kwargs)
        self._base_observation_dim = int(np.asarray(self.state, dtype=np.float32).size)
        if self.observation_history_window > 1:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._base_observation_dim * self.observation_history_window,),
                dtype=np.float32,
            )
        self.historical_window = 100

    def _scale_obs_cash(self, obs):
        """Scale only the cash observation returned to the agent."""
        if self.cash_scale_factor == 1.0 or obs is None:
            return obs
        if isinstance(obs, np.ndarray):
            out = obs.copy()
            if out.size > 0:
                out[0] = out[0] / self.cash_scale_factor
            return out
        out = list(obs)
        if out:
            out[0] = float(out[0]) / self.cash_scale_factor
        return out

    def _as_obs_array(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _reset_history_obs(self, obs):
        scaled = self._as_obs_array(self._scale_obs_cash(obs))
        self._obs_history = [scaled.copy() for _ in range(self.observation_history_window)]
        if self.observation_history_window == 1:
            return scaled
        return np.concatenate(self._obs_history, axis=0).astype(np.float32, copy=False)

    def _append_history_obs(self, obs):
        scaled = self._as_obs_array(self._scale_obs_cash(obs))
        if not self._obs_history:
            self._obs_history = [scaled.copy() for _ in range(self.observation_history_window)]
        else:
            self._obs_history.append(scaled.copy())
            self._obs_history = self._obs_history[-self.observation_history_window :]
            while len(self._obs_history) < self.observation_history_window:
                self._obs_history.insert(0, self._obs_history[0].copy())
        if self.observation_history_window == 1:
            return scaled
        return np.concatenate(self._obs_history, axis=0).astype(np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        self._simple_mcts_pending_snapshots = {}
        reset_out = super().reset(seed=seed, options=options)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
            return self._reset_history_obs(obs), info
        return self._reset_history_obs(reset_out)

    def _sharpe_like_from_returns(self, returns: np.ndarray) -> float:
        """Compute a clipped Sharpe-like reward from daily returns."""
        returns = np.asarray(returns, dtype=np.float64).ravel()
        if returns.size < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        if not np.isfinite(mean_r) or not np.isfinite(std_r):
            return 0.0
        denom = max(std_r, 1e-4)
        r = mean_r / denom
        return float(np.clip(r, -50.0, 50.0))

    def _compute_lookahead_reward(
        self, current_day: int, cash: float, shares: np.ndarray
    ) -> float:
        """Compute the look-ahead reward without changing the current position."""
        unique_days = self.df.index.unique()
        last_idx = len(unique_days) - 1

        end_day = min(current_day + self.lookahead_window, last_idx)

        if end_day <= current_day:
            return 0.0

        asset_values = []
        dates = []

        for day_idx in range(current_day, end_day + 1):
            data_day = self.df.loc[day_idx, :]
            day_date = None
            if isinstance(data_day, pd.DataFrame):
                if "date" in data_day.columns and len(data_day) > 0:
                    day_date = data_day["date"].iloc[0]
                if "close" in data_day.columns:
                    prices = data_day["close"].values
                else:
                    prices = np.zeros(self.stock_dim, dtype=np.float64)
            else:
                if hasattr(data_day, "get"):
                    day_date = data_day.get("date", None)
                    close_val = data_day.get("close", None)
                else:
                    day_date = None
                    close_val = None
                prices = np.array([close_val], dtype=np.float64) if close_val is not None else np.zeros(1, dtype=np.float64)

            total_asset = float(cash + np.sum(shares * prices))
            asset_values.append(total_asset)
            dates.append(day_date)

        asset_values = np.asarray(asset_values, dtype=np.float64)

        returns = asset_values[1:] / (asset_values[:-1] + 1e-8) - 1.0
        if returns.size == 0:
            return 0.0

        reward = self._sharpe_like_from_returns(returns)

        return reward

    def _compute_simple_mcts_window_reward(
        self,
        window_start: int,
        window_end: int,
        cash: float,
        shares: np.ndarray,
    ) -> float:
        """Compute a delayed reward from an observed price window."""
        unique_days = self.df.index.unique()
        last_idx = len(unique_days) - 1
        window_start = int(window_start)
        window_end = min(int(window_end), last_idx)
        if window_end <= window_start:
            return 0.0

        asset_values = []
        for day_idx in range(window_start, window_end + 1):
            data_day = self.df.loc[day_idx, :]

            if isinstance(data_day, pd.DataFrame):
                if "close" in data_day.columns:
                    prices = data_day["close"].values
                else:
                    prices = np.zeros(self.stock_dim, dtype=np.float64)
            else:
                close_val = data_day.get("close", None) if hasattr(data_day, "get") else None
                prices = (
                    np.array([close_val], dtype=np.float64)
                    if close_val is not None
                    else np.zeros(1, dtype=np.float64)
                )

            total_asset = float(cash + np.sum(shares * prices))
            asset_values.append(total_asset)

        asset_values = np.asarray(asset_values, dtype=np.float64)
        if len(asset_values) < 2:
            return 0.0

        returns = asset_values[1:] / (asset_values[:-1] + 1e-8) - 1.0
        if returns.size == 0:
            return 0.0
        return self._sharpe_like_from_returns(returns)

    def _compute_simple_mcts_reward(
        self, current_day: int, cash: float, shares: np.ndarray
    ) -> float:
        """Compute the auxiliary window reward when delay is disabled."""
        if self.simple_mcts_reward_delay_steps > 0:
            return 0.0

        total_reward = 0.0
        for window_offset in range(5):
            total_reward += self._compute_simple_mcts_window_reward(
                window_start=current_day + window_offset,
                window_end=current_day + window_offset + self.lookahead_window,
                cash=cash,
                shares=shares,
            )
        return total_reward

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            step_out = super().step(actions)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                return self._append_history_obs(obs), reward, terminated, truncated, info
            return step_out

        current_day = self.day

        actions = actions * self.hmax
        actions = actions.astype(int)

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-self.hmax] * self.stock_dim)

        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        self.actions_memory.append(actions)

        self.day += 1
        self.data = self.df.loc[self.day, :]

        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif len(self.df.tic.unique()) > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]

        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        immediate_reward = end_total_asset - begin_total_asset
        
        cash = float(self.state[0])
        shares = np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        
        simple_mcts_reward = 0.0
        if self.enable_simple_mcts:
            simple_mcts_reward = self._compute_simple_mcts_reward(
                current_day=current_day, cash=cash, shares=shares
            )
        
        if self.use_lookahead_reward:
            lookahead_reward = self._compute_lookahead_reward(
                current_day=current_day, cash=cash, shares=shares
            )
            base_reward = lookahead_reward
            lookahead_reward_used = lookahead_reward
            simple_mcts_reward_used = simple_mcts_reward
        else:
            base_reward = immediate_reward
            lookahead_reward_used = None
            simple_mcts_reward_used = simple_mcts_reward

        if self.simple_mcts_reward_delay_steps > 0 and self.enable_simple_mcts:
            mcts_in_return = 0.0
            due_snapshots = self._simple_mcts_pending_snapshots.pop(current_day, [])
            for start_day, start_cash, start_shares in due_snapshots:
                mcts_in_return += self._compute_simple_mcts_window_reward(
                    window_start=start_day,
                    window_end=current_day,
                    cash=start_cash,
                    shares=start_shares,
                )

            emit_key = current_day + self.simple_mcts_reward_delay_steps
            self._simple_mcts_pending_snapshots.setdefault(emit_key, []).append(
                (current_day, cash, shares.astype(np.float64, copy=True))
            )
        else:
            mcts_in_return = float(simple_mcts_reward)

        final_reward = base_reward + mcts_in_return
        reward_out = base_reward + mcts_in_return

        self.rewards_memory.append(reward_out)

        self.reward = reward_out * self.reward_scaling
        scaled_state = self._scale_obs_cash(self.state)
        self.state_memory.append(scaled_state)

        return self._append_history_obs(self.state), self.reward, self.terminal, False, {
            "immediate_reward": immediate_reward,
            "lookahead_reward": lookahead_reward_used,
            "simple_mcts_reward": simple_mcts_reward_used,
            "final_reward": final_reward,
            "reward_returned": reward_out,
            "simple_mcts_reward_in_return": mcts_in_return,
            "simple_mcts_reward_delay_steps": self.simple_mcts_reward_delay_steps,
            "cash_scale_factor": self.cash_scale_factor,
        }



