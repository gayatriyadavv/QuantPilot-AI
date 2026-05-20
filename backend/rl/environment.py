"""
QuantPilot AI — Custom Gymnasium Trading Environment

A fully-featured trading environment compatible with Stable-Baselines3.
Supports discrete actions (Hold/Buy/Sell), transaction costs, slippage,
and composite reward functions.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnv(gym.Env):
    """Custom Gymnasium environment for algorithmic trading.

    Observation Space:
        Box containing a flattened window of:
        - Normalized OHLCV data
        - Technical indicators
        - Portfolio state (cash ratio, position ratio, unrealized PnL ratio)

    Action Space:
        Discrete(3): 0 = Hold, 1 = Buy, 2 = Sell

    Reward:
        Configurable — defaults to change in portfolio value minus costs.
    """

    metadata = {"render_modes": ["human"]}

    # Action constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 100_000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
        max_position_pct: float = 1.0,
        reward_function: Optional[Any] = None,
        features: Optional[list[str]] = None,
        normalize_obs: bool = True,
    ):
        """Initialize the trading environment.

        Args:
            df: OHLCV DataFrame with optional indicator columns
            window_size: Number of past bars in observation
            initial_balance: Starting cash balance
            commission_rate: Trading commission as fraction
            slippage: Slippage as fraction of price
            max_position_pct: Max fraction of portfolio for single position
            reward_function: Custom reward callable(env) → float
            features: List of columns to include in observation (default: all numeric)
            normalize_obs: Whether to normalize observation values
        """
        super().__init__()

        self.df = df.copy()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.reward_fn = reward_function
        self.normalize_obs = normalize_obs

        # Determine feature columns
        if features is not None:
            self.feature_columns = [f for f in features if f in df.columns]
        else:
            self.feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.n_features = len(self.feature_columns)
        self.n_portfolio_features = 3  # cash_ratio, position_ratio, unrealized_pnl_ratio

        # Precompute normalized data
        self._data = df[self.feature_columns].values.astype(np.float32)
        if self.normalize_obs:
            self._data_mean = np.nanmean(self._data, axis=0)
            self._data_std = np.nanstd(self._data, axis=0) + 1e-8
            self._data_normalized = (self._data - self._data_mean) / self._data_std
        else:
            self._data_normalized = self._data

        self._prices = df["close"].values.astype(np.float64)
        self.max_steps = len(df) - 1

        # ── Spaces ──
        obs_shape = (self.window_size * self.n_features + self.n_portfolio_features,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # ── State Variables (initialized in reset) ──
        self.current_step: int = 0
        self.cash: float = 0.0
        self.shares: float = 0.0
        self.entry_price: float = 0.0
        self.total_value: float = 0.0
        self.prev_total_value: float = 0.0
        self.trade_count: int = 0
        self.trade_history: list[dict] = []

        logger.info(
            f"TradingEnv initialized: {len(df)} bars, {self.n_features} features, "
            f"window={window_size}, balance=${initial_balance:,.0f}"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.shares = 0.0
        self.entry_price = 0.0
        self.total_value = self.initial_balance
        self.prev_total_value = self.initial_balance
        self.trade_count = 0
        self.trade_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.prev_total_value = self.total_value
        current_price = self._prices[self.current_step]

        # Execute action
        trade_info = self._execute_action(action, current_price)

        # Update portfolio value
        self.total_value = self.cash + self.shares * current_price

        # Calculate reward
        if self.reward_fn is not None:
            reward = self.reward_fn(self)
        else:
            reward = self._default_reward()

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = (
            self.current_step >= self.max_steps or
            self.total_value <= self.initial_balance * 0.5  # 50% loss → stop
        )
        truncated = False

        obs = self._get_observation()
        info = self._get_info()
        info["trade"] = trade_info

        return obs, float(reward), terminated, truncated, info

    # ── Action Execution ──────────────────────────────────────

    def _execute_action(self, action: int, price: float) -> dict:
        """Execute a trade action with transaction costs.

        Returns trade info dict.
        """
        trade_info = {"action": action, "executed": False, "price": price}

        if action == self.BUY and self.shares == 0:
            # Buy with slippage
            fill_price = price * (1 + self.slippage)
            max_invest = self.cash * self.max_position_pct
            commission = max_invest * self.commission_rate
            invest_amount = max_invest - commission
            shares_to_buy = invest_amount / fill_price

            if shares_to_buy > 0:
                self.shares = shares_to_buy
                self.entry_price = fill_price
                self.cash -= (shares_to_buy * fill_price + commission)
                self.trade_count += 1

                trade_info.update({
                    "executed": True,
                    "side": "BUY",
                    "shares": shares_to_buy,
                    "fill_price": fill_price,
                    "commission": commission,
                })

        elif action == self.SELL and self.shares > 0:
            # Sell with slippage
            fill_price = price * (1 - self.slippage)
            proceeds = self.shares * fill_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission
            pnl = net_proceeds - (self.shares * self.entry_price)

            self.cash += net_proceeds
            self.trade_count += 1

            trade_info.update({
                "executed": True,
                "side": "SELL",
                "shares": self.shares,
                "fill_price": fill_price,
                "commission": commission,
                "pnl": pnl,
                "pnl_pct": pnl / (self.shares * self.entry_price) if self.entry_price > 0 else 0,
            })

            self.trade_history.append(trade_info.copy())
            self.shares = 0.0
            self.entry_price = 0.0

        return trade_info

    # ── Observation ───────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector.

        Combines:
        1. Window of normalized market data
        2. Portfolio state features
        """
        # Market data window
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        window_data = self._data_normalized[start:end].flatten()

        # Pad if window is shorter than expected (at start)
        expected_len = self.window_size * self.n_features
        if len(window_data) < expected_len:
            padding = np.zeros(expected_len - len(window_data), dtype=np.float32)
            window_data = np.concatenate([padding, window_data])

        # Portfolio features
        current_price = self._prices[min(self.current_step, self.max_steps)]
        position_value = self.shares * current_price
        total = self.cash + position_value

        portfolio_features = np.array([
            self.cash / max(total, 1e-8),           # cash ratio
            position_value / max(total, 1e-8),      # position ratio
            (total - self.initial_balance) / self.initial_balance,  # unrealized PnL ratio
        ], dtype=np.float32)

        obs = np.concatenate([window_data, portfolio_features])
        return obs.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────

    def _default_reward(self) -> float:
        """Default reward: log return of portfolio value."""
        if self.prev_total_value <= 0:
            return 0.0
        return float(np.log(self.total_value / self.prev_total_value))

    # ── Info ──────────────────────────────────────────────────

    def _get_info(self) -> dict:
        """Return current environment info."""
        current_price = self._prices[min(self.current_step, self.max_steps)]
        return {
            "step": self.current_step,
            "cash": self.cash,
            "shares": self.shares,
            "position_value": self.shares * current_price,
            "total_value": self.total_value,
            "portfolio_return": (self.total_value - self.initial_balance) / self.initial_balance,
            "trade_count": self.trade_count,
            "current_price": current_price,
        }

    # ── Render ────────────────────────────────────────────────

    def render(self):
        """Print current state."""
        info = self._get_info()
        print(
            f"Step {info['step']}/{self.max_steps} | "
            f"Value: ${info['total_value']:,.2f} | "
            f"Return: {info['portfolio_return']:.2%} | "
            f"Cash: ${info['cash']:,.2f} | "
            f"Shares: {info['shares']:.4f} | "
            f"Trades: {info['trade_count']}"
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def portfolio_return(self) -> float:
        """Current portfolio return."""
        return (self.total_value - self.initial_balance) / self.initial_balance

    @property
    def current_price(self) -> float:
        """Current asset price."""
        return float(self._prices[min(self.current_step, self.max_steps)])

    @property
    def is_holding(self) -> bool:
        """Whether a position is currently held."""
        return self.shares > 0

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized PnL of current position."""
        if not self.is_holding:
            return 0.0
        return self.shares * (self.current_price - self.entry_price)


def create_trading_env(
    df: pd.DataFrame,
    window_size: int = 30,
    initial_balance: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    reward_function: Optional[Any] = None,
    **kwargs,
) -> TradingEnv:
    """Factory function to create a TradingEnv.

    Convenience wrapper that also validates the environment.
    """
    env = TradingEnv(
        df=df,
        window_size=window_size,
        initial_balance=initial_balance,
        commission_rate=commission,
        slippage=slippage,
        reward_function=reward_function,
        **kwargs,
    )

    # Validate
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        logger.info("Environment validation passed ✅")
    except Exception as e:
        logger.warning(f"Environment validation warning: {e}")

    return env
