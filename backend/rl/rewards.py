"""
QuantPilot AI — Modular Reward Functions

Composable reward functions for the RL trading environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger


class RewardFunction(ABC):
    """Abstract base for reward functions."""

    @abstractmethod
    def calculate(self, env) -> float:
        """Calculate reward given current environment state.

        Args:
            env: TradingEnv instance with current state

        Returns:
            Scalar reward value
        """
        ...

    def reset(self):
        """Reset internal state for new episode."""
        pass

    def __call__(self, env) -> float:
        return self.calculate(env)


class ProfitReward(RewardFunction):
    """Simple PnL-based reward: change in portfolio value."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def calculate(self, env) -> float:
        if env.prev_total_value <= 0:
            return 0.0
        pnl_pct = (env.total_value - env.prev_total_value) / env.prev_total_value
        return float(pnl_pct * self.scale)


class LogReturnReward(RewardFunction):
    """Log return reward — better for multiplicative processes."""

    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def calculate(self, env) -> float:
        if env.prev_total_value <= 0:
            return 0.0
        return float(np.log(env.total_value / env.prev_total_value) * self.scale)


class SharpeReward(RewardFunction):
    """Rolling Sharpe ratio reward — encourages consistent risk-adjusted returns."""

    def __init__(self, window: int = 30, risk_free_rate: float = 0.04, scale: float = 1.0):
        self.window = window
        self.daily_rfr = (1 + risk_free_rate) ** (1 / 252) - 1
        self.scale = scale
        self._returns: deque = deque(maxlen=window)

    def reset(self):
        self._returns.clear()

    def calculate(self, env) -> float:
        if env.prev_total_value <= 0:
            return 0.0

        ret = (env.total_value - env.prev_total_value) / env.prev_total_value
        self._returns.append(ret)

        if len(self._returns) < 5:
            return float(ret * self.scale)

        returns = np.array(self._returns)
        excess = returns - self.daily_rfr
        std = returns.std()

        if std < 1e-8:
            return 0.0

        sharpe = excess.mean() / std
        return float(sharpe * self.scale)


class DrawdownPenalty(RewardFunction):
    """Penalize drawdowns from peak portfolio value."""

    def __init__(self, penalty_factor: float = 2.0, threshold: float = 0.05):
        """
        Args:
            penalty_factor: Multiplier for drawdown penalty
            threshold: Minimum drawdown to trigger penalty (e.g., 5%)
        """
        self.penalty_factor = penalty_factor
        self.threshold = threshold
        self._peak_value: float = 0.0

    def reset(self):
        self._peak_value = 0.0

    def calculate(self, env) -> float:
        self._peak_value = max(self._peak_value, env.total_value)

        if self._peak_value <= 0:
            return 0.0

        drawdown = (self._peak_value - env.total_value) / self._peak_value

        if drawdown > self.threshold:
            return float(-drawdown * self.penalty_factor)

        return 0.0


class CostAwareReward(RewardFunction):
    """Reward that explicitly penalizes transaction costs and excessive trading."""

    def __init__(
        self,
        base_scale: float = 100.0,
        trade_penalty: float = 0.001,
        hold_bonus: float = 0.0001,
    ):
        self.base_scale = base_scale
        self.trade_penalty = trade_penalty
        self.hold_bonus = hold_bonus
        self._prev_trade_count: int = 0

    def reset(self):
        self._prev_trade_count = 0

    def calculate(self, env) -> float:
        if env.prev_total_value <= 0:
            return 0.0

        # Base reward: log return
        base = np.log(env.total_value / env.prev_total_value) * self.base_scale

        # Penalty for trading
        traded = env.trade_count > self._prev_trade_count
        self._prev_trade_count = env.trade_count

        if traded:
            reward = base - self.trade_penalty
        elif env.is_holding:
            # Small bonus for holding profitable positions
            if env.unrealized_pnl > 0:
                reward = base + self.hold_bonus
            else:
                reward = base
        else:
            reward = base

        return float(reward)


class CompositeReward(RewardFunction):
    """Weighted combination of multiple reward functions.

    Usage:
        reward = CompositeReward([
            (ProfitReward(), 0.4),
            (SharpeReward(), 0.3),
            (DrawdownPenalty(), 0.2),
            (CostAwareReward(), 0.1),
        ])
    """

    def __init__(self, components: list[tuple[RewardFunction, float]]):
        """
        Args:
            components: List of (RewardFunction, weight) tuples
        """
        self.components = components
        total_weight = sum(w for _, w in components)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Composite reward weights sum to {total_weight:.3f}, not 1.0")

    def reset(self):
        for func, _ in self.components:
            func.reset()

    def calculate(self, env) -> float:
        total = 0.0
        for func, weight in self.components:
            total += func.calculate(env) * weight
        return float(total)


# ── Preset Reward Configurations ──────────────────────────────


def get_default_reward() -> CompositeReward:
    """Get the default composite reward function."""
    return CompositeReward([
        (LogReturnReward(scale=100.0), 0.4),
        (SharpeReward(window=30), 0.3),
        (DrawdownPenalty(penalty_factor=2.0, threshold=0.05), 0.2),
        (CostAwareReward(trade_penalty=0.001), 0.1),
    ])


def get_aggressive_reward() -> CompositeReward:
    """Reward that prioritizes absolute returns."""
    return CompositeReward([
        (ProfitReward(scale=100.0), 0.6),
        (SharpeReward(window=20), 0.2),
        (DrawdownPenalty(penalty_factor=1.0, threshold=0.10), 0.2),
    ])


def get_conservative_reward() -> CompositeReward:
    """Reward that prioritizes risk-adjusted returns and drawdown control."""
    return CompositeReward([
        (LogReturnReward(scale=50.0), 0.2),
        (SharpeReward(window=50), 0.4),
        (DrawdownPenalty(penalty_factor=5.0, threshold=0.03), 0.3),
        (CostAwareReward(trade_penalty=0.002), 0.1),
    ])
