"""
QuantPilot AI — RL Training Pipeline

End-to-end training pipeline: data → indicators → environment → agent → evaluation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backend.config import CHECKPOINTS_DIR, RLAlgorithm, get_settings
from backend.data.fetcher import DataFetcherFactory
from backend.data.processor import DataProcessor
from backend.indicators.technical import TechnicalIndicators
from backend.rl.agents import AgentFactory, TradingAgent
from backend.rl.callbacks import (
    BestModelCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    TradingMetricsCallback,
)
from backend.rl.environment import TradingEnv, create_trading_env
from backend.rl.rewards import get_default_reward


class TrainingPipeline:
    """Complete training pipeline for RL trading agents.

    Handles data fetching, preprocessing, environment creation,
    training, evaluation, and model persistence.
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        total_timesteps: int = 100_000,
        window_size: int = 30,
        initial_balance: float = 100_000.0,
        commission: float = 0.001,
        data_period: str = "2y",
        train_ratio: float = 0.8,
        model_name: Optional[str] = None,
        hyperparams: Optional[dict] = None,
    ):
        self.symbol = symbol
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.data_period = data_period
        self.train_ratio = train_ratio
        self.model_name = model_name or f"{algorithm.value}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.hyperparams = hyperparams

        # Components
        self.processor = DataProcessor()
        self.indicator_calc = TechnicalIndicators()

        # State
        self.raw_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.agent: Optional[TradingAgent] = None
        self.training_results: Optional[dict] = None

    async def fetch_data(self) -> pd.DataFrame:
        """Step 1: Fetch market data."""
        logger.info(f"Fetching {self.symbol} data (period={self.data_period})")
        fetcher = DataFetcherFactory.auto_detect(self.symbol)
        self.raw_data = await fetcher.fetch_ohlcv(self.symbol, period=self.data_period)

        if self.raw_data.empty:
            raise ValueError(f"No data available for {self.symbol}")

        logger.info(f"Fetched {len(self.raw_data)} bars")
        return self.raw_data

    def prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Step 2: Add indicators and split data."""
        if self.raw_data is None:
            raise ValueError("Call fetch_data() first")

        # Handle missing values
        data = self.processor.handle_missing_values(self.raw_data)

        # Add indicators
        data = self.indicator_calc.add_all_indicators(data)

        # Split chronologically
        split_idx = int(len(data) * self.train_ratio)
        self.train_data = data.iloc[:split_idx].copy()
        self.test_data = data.iloc[split_idx:].copy()

        logger.info(f"Data split: train={len(self.train_data)}, test={len(self.test_data)}")
        return self.train_data, self.test_data

    def create_environments(self) -> tuple[TradingEnv, TradingEnv]:
        """Step 3: Create training and evaluation environments."""
        if self.train_data is None:
            raise ValueError("Call prepare_data() first")

        reward_fn = get_default_reward()

        train_env = TradingEnv(
            df=self.train_data,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            commission_rate=self.commission,
            reward_function=reward_fn,
        )

        test_env = TradingEnv(
            df=self.test_data,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            commission_rate=self.commission,
        )

        return train_env, test_env

    def train(self, train_env: TradingEnv) -> TradingAgent:
        """Step 4: Train the RL agent."""
        self.agent = AgentFactory.create(
            algorithm=self.algorithm,
            env=train_env,
            hyperparams=self.hyperparams,
            model_name=self.model_name,
        )

        # Setup callbacks
        callbacks = [
            TradingMetricsCallback(log_interval=5000),
            EarlyStoppingCallback(max_drawdown=0.3, patience=20),
            BestModelCallback(
                save_dir=str(CHECKPOINTS_DIR / self.model_name),
                check_interval=10_000,
            ),
            ProgressCallback(self.total_timesteps),
        ]

        logger.info(f"Starting training: {self.algorithm.value} for {self.total_timesteps:,} steps")
        start = time.time()

        self.agent.train(
            total_timesteps=self.total_timesteps,
            callbacks=callbacks,
            progress_bar=False,
        )

        elapsed = time.time() - start
        logger.info(f"Training completed in {elapsed:.1f}s")

        return self.agent

    def evaluate(self, test_env: TradingEnv) -> dict:
        """Step 5: Evaluate the trained agent."""
        if self.agent is None:
            raise ValueError("Call train() first")

        results = self.agent.evaluate(test_env, n_episodes=3)
        self.training_results = results
        return results

    def save_model(self) -> str:
        """Step 6: Save the trained model."""
        if self.agent is None:
            raise ValueError("No trained agent to save")

        path = self.agent.save(str(CHECKPOINTS_DIR))
        logger.info(f"Model saved: {path}")
        return path

    async def run(self) -> dict:
        """Execute the full training pipeline.

        Returns:
            Summary dict with training and evaluation results.
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"  QuantPilot AI — Training Pipeline")
        logger.info(f"  Symbol: {self.symbol} | Algorithm: {self.algorithm.value}")
        logger.info(f"  Steps: {self.total_timesteps:,} | Window: {self.window_size}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # 1. Fetch data
        await self.fetch_data()

        # 2. Prepare data
        self.prepare_data()

        # 3. Create environments
        train_env, test_env = self.create_environments()

        # 4. Train
        self.train(train_env)

        # 5. Evaluate
        eval_results = self.evaluate(test_env)

        # 6. Save
        model_path = self.save_model()

        elapsed = time.time() - start_time

        summary = {
            "symbol": self.symbol,
            "algorithm": self.algorithm.value,
            "model_name": self.model_name,
            "model_path": model_path,
            "total_timesteps": self.total_timesteps,
            "training_time_secs": elapsed,
            "train_data_size": len(self.train_data),
            "test_data_size": len(self.test_data),
            "evaluation": eval_results,
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Training Complete!")
        logger.info(f"  Time: {elapsed:.1f}s")
        logger.info(f"  Eval Return: {eval_results.get('mean_return', 0):.2%}")
        logger.info(f"  Model: {model_path}")
        logger.info(f"{'=' * 60}")

        return summary


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantPilot AI — Train RL Agent")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A2C"])
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--window", type=int, default=30, help="Observation window size")
    parser.add_argument("--balance", type=float, default=100_000, help="Initial balance")
    parser.add_argument("--period", type=str, default="2y", help="Data period")
    args = parser.parse_args()

    async def main():
        pipeline = TrainingPipeline(
            symbol=args.symbol,
            algorithm=RLAlgorithm(args.algorithm),
            total_timesteps=args.timesteps,
            window_size=args.window,
            initial_balance=args.balance,
            data_period=args.period,
        )
        results = await pipeline.run()
        print(f"\n✅ Training complete. Model: {results['model_path']}")

    asyncio.run(main())
