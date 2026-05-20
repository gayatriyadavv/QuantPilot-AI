"""
QuantPilot AI — Custom Stable-Baselines3 Callbacks

Trading-specific callbacks for training monitoring, early stopping, and checkpointing.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    """Log trading-specific metrics during training.

    Tracks portfolio value, returns, trade count, and win rate at intervals.
    """

    def __init__(
        self,
        log_interval: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_rewards: list[float] = []
        self._episode_returns: list[float] = []
        self._episode_trades: list[int] = []

    def _on_step(self) -> bool:
        # Check for episode completion
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

            if "portfolio_return" in info:
                self._episode_returns.append(info["portfolio_return"])
            if "trade_count" in info:
                self._episode_trades.append(info["trade_count"])

        # Log at intervals
        if self.n_calls % self.log_interval == 0 and self.verbose > 0:
            if self._episode_returns:
                mean_return = np.mean(self._episode_returns[-10:])
                mean_trades = np.mean(self._episode_trades[-10:]) if self._episode_trades else 0

                logger.info(
                    f"Step {self.num_timesteps:>8,d} | "
                    f"Mean Return: {mean_return:.2%} | "
                    f"Mean Trades: {mean_trades:.0f} | "
                    f"Episodes: {len(self._episode_returns)}"
                )

                # Log to TensorBoard if available
                if self.logger:
                    self.logger.record("trading/mean_return", mean_return)
                    self.logger.record("trading/mean_trades", mean_trades)
                    self.logger.record("trading/episodes", len(self._episode_returns))

        return True

    def _on_training_end(self) -> None:
        if self._episode_returns:
            logger.info(
                f"Training finished | "
                f"Final Mean Return: {np.mean(self._episode_returns[-20:]):.2%} | "
                f"Total Episodes: {len(self._episode_returns)}"
            )


class EarlyStoppingCallback(BaseCallback):
    """Stop training if portfolio drawdown exceeds threshold.

    Also stops if mean reward hasn't improved for `patience` evaluations.
    """

    def __init__(
        self,
        max_drawdown: float = 0.3,
        patience: int = 20,
        min_improvement: float = 0.01,
        check_interval: int = 5000,
        verbose: int = 1,
    ):
        """
        Args:
            max_drawdown: Maximum portfolio drawdown before stopping
            patience: Number of checks without improvement before stopping
            min_improvement: Minimum reward improvement to reset patience
            check_interval: Steps between checks
        """
        super().__init__(verbose)
        self.max_drawdown = max_drawdown
        self.patience = patience
        self.min_improvement = min_improvement
        self.check_interval = check_interval
        self._best_mean_reward = -np.inf
        self._no_improvement_count = 0
        self._rewards_buffer: deque = deque(maxlen=50)

    def _on_step(self) -> bool:
        # Collect rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._rewards_buffer.append(info["episode"]["r"])

        # Check at intervals
        if self.n_calls % self.check_interval == 0 and len(self._rewards_buffer) > 5:
            mean_reward = np.mean(list(self._rewards_buffer))

            # Check improvement
            if mean_reward > self._best_mean_reward + self.min_improvement:
                self._best_mean_reward = mean_reward
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1

            # Patience exhausted
            if self._no_improvement_count >= self.patience:
                if self.verbose > 0:
                    logger.warning(
                        f"Early stopping: no improvement for "
                        f"{self.patience * self.check_interval:,} steps"
                    )
                return False

        # Check drawdown from infos
        for info in infos:
            portfolio_return = info.get("portfolio_return", 0)
            if portfolio_return < -self.max_drawdown:
                if self.verbose > 0:
                    logger.warning(
                        f"Early stopping: drawdown {portfolio_return:.2%} "
                        f"exceeds limit {-self.max_drawdown:.2%}"
                    )
                # Don't stop training, just flag — let episode end naturally

        return True


class BestModelCallback(BaseCallback):
    """Save the model with the best rolling performance."""

    def __init__(
        self,
        save_dir: str,
        check_interval: int = 10_000,
        metric: str = "reward",  # 'reward' or 'return'
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.check_interval = check_interval
        self.metric = metric
        self._best_value = -np.inf
        self._rewards: deque = deque(maxlen=30)
        self._returns: deque = deque(maxlen=30)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._rewards.append(info["episode"]["r"])
            if "portfolio_return" in info:
                self._returns.append(info["portfolio_return"])

        if self.n_calls % self.check_interval == 0:
            buffer = self._rewards if self.metric == "reward" else self._returns
            if len(buffer) >= 5:
                current_value = np.mean(list(buffer))

                if current_value > self._best_value:
                    self._best_value = current_value
                    self._save_best()

                    if self.verbose > 0:
                        logger.info(
                            f"New best model (step {self.num_timesteps:,}): "
                            f"{self.metric}={current_value:.4f}"
                        )

        return True

    def _save_best(self):
        """Save the current model as the best."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = str(self.save_dir / "best_model")
        self.model.save(path)


class ProgressCallback(BaseCallback):
    """Simple progress logging callback."""

    def __init__(self, total_timesteps: int, log_interval: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            logger.info(f"Training progress: {progress:.1f}% ({self.num_timesteps:,}/{self.total_timesteps:,})")
        return True
