"""
QuantPilot AI — RL Agent Wrappers

Unified interface for PPO, DQN, A2C agents using Stable-Baselines3.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
from loguru import logger
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backend.config import CHECKPOINTS_DIR, RLAlgorithm
from backend.rl.environment import TradingEnv


# ── Default Hyperparameters ───────────────────────────────────

DEFAULT_HYPERPARAMS = {
    RLAlgorithm.PPO: {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    RLAlgorithm.DQN: {
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
    RLAlgorithm.A2C: {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "policy_kwargs": {"net_arch": [256, 256]},
    },
}

ALGO_CLASS_MAP: dict[RLAlgorithm, Type[BaseAlgorithm]] = {
    RLAlgorithm.PPO: PPO,
    RLAlgorithm.DQN: DQN,
    RLAlgorithm.A2C: A2C,
}


# ── Trading Agent ─────────────────────────────────────────────


class TradingAgent:
    """Unified wrapper for RL trading agents.

    Handles training, prediction, saving, loading, and evaluation
    with a consistent interface across PPO, DQN, and A2C.
    """

    def __init__(
        self,
        algorithm: RLAlgorithm,
        env: TradingEnv,
        hyperparams: Optional[dict] = None,
        model_name: Optional[str] = None,
        use_vec_normalize: bool = True,
        tensorboard_log: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize the trading agent.

        Args:
            algorithm: PPO, DQN, or A2C
            env: Trading environment
            hyperparams: Override default hyperparameters
            model_name: Name for saving/loading
            use_vec_normalize: Whether to normalize observations/rewards
            tensorboard_log: Path for TensorBoard logs
            device: 'auto', 'cpu', or 'cuda'
        """
        self.algorithm = algorithm
        self.model_name = model_name or f"{algorithm.value}_trading"
        self.use_vec_normalize = use_vec_normalize

        # Setup vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        if use_vec_normalize:
            self.vec_env = VecNormalize(
                self.vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )

        # Merge hyperparameters
        params = DEFAULT_HYPERPARAMS.get(algorithm, {}).copy()
        if hyperparams:
            params.update(hyperparams)

        # Create model
        algo_class = ALGO_CLASS_MAP[algorithm]
        self.model = algo_class(
            "MlpPolicy",
            self.vec_env,
            verbose=0,
            tensorboard_log=tensorboard_log,
            device=device,
            **params,
        )

        self._is_trained = False
        logger.info(
            f"Agent created: {algorithm.value} | "
            f"device={device} | vec_normalize={use_vec_normalize}"
        )

    def train(
        self,
        total_timesteps: int = 100_000,
        callbacks: Optional[list[BaseCallback]] = None,
        progress_bar: bool = True,
        log_interval: int = 10,
    ) -> dict:
        """Train the agent.

        Args:
            total_timesteps: Number of training timesteps
            callbacks: List of SB3 callbacks
            progress_bar: Show training progress bar
            log_interval: Logging frequency (in episodes)

        Returns:
            Training summary dict
        """
        logger.info(f"Training {self.algorithm.value} for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
            log_interval=log_interval,
        )

        self._is_trained = True
        logger.info(f"Training complete: {total_timesteps:,} timesteps")

        return {
            "algorithm": self.algorithm.value,
            "total_timesteps": total_timesteps,
            "model_name": self.model_name,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[int, Optional[np.ndarray]]:
        """Get action prediction for a single observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            (action, action_probabilities)
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return int(action), _states

    def evaluate(
        self,
        env: TradingEnv,
        n_episodes: int = 5,
    ) -> dict:
        """Evaluate agent on an environment.

        Args:
            env: Evaluation environment
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics dict
        """
        eval_env = DummyVecEnv([lambda: env])
        if self.use_vec_normalize:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
            if isinstance(self.vec_env, VecNormalize):
                eval_env.obs_rms = self.vec_env.obs_rms

        episode_rewards = []
        episode_returns = []
        episode_trades = []

        for ep in range(n_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += float(reward[0])

                if done[0]:
                    episode_rewards.append(total_reward)
                    info_dict = info[0]
                    episode_returns.append(info_dict.get("portfolio_return", 0))
                    episode_trades.append(info_dict.get("trade_count", 0))

        results = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0,
            "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0,
            "mean_return": float(np.mean(episode_returns)) if episode_returns else 0,
            "mean_trades": float(np.mean(episode_trades)) if episode_trades else 0,
            "n_episodes": n_episodes,
        }

        logger.info(
            f"Evaluation: reward={results['mean_reward']:.2f}±{results['std_reward']:.2f}, "
            f"return={results['mean_return']:.2%}, trades={results['mean_trades']:.0f}"
        )

        return results

    def save(self, directory: Optional[str] = None) -> str:
        """Save the model and normalizer to disk.

        Args:
            directory: Save directory (default: checkpoints)

        Returns:
            Path to saved model
        """
        save_dir = Path(directory or CHECKPOINTS_DIR) / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(save_dir / "model")
        self.model.save(model_path)

        if self.use_vec_normalize and isinstance(self.vec_env, VecNormalize):
            norm_path = str(save_dir / "vec_normalize.pkl")
            self.vec_env.save(norm_path)

        logger.info(f"Model saved to {save_dir}")
        return str(save_dir)

    @classmethod
    def load(
        cls,
        model_path: str,
        env: TradingEnv,
        algorithm: Optional[RLAlgorithm] = None,
        device: str = "auto",
    ) -> "TradingAgent":
        """Load a saved model.

        Args:
            model_path: Path to model directory
            env: Environment for the loaded model
            algorithm: Algorithm type (auto-detected if None)
            device: Device for inference

        Returns:
            Loaded TradingAgent
        """
        model_dir = Path(model_path)
        model_file = model_dir / "model.zip"

        if not model_file.exists():
            # Try without .zip
            model_file = model_dir / "model"

        # Auto-detect algorithm from directory name
        if algorithm is None:
            dir_name = model_dir.name.upper()
            for algo in RLAlgorithm:
                if algo.value in dir_name:
                    algorithm = algo
                    break
            if algorithm is None:
                algorithm = RLAlgorithm.PPO  # Default

        # Create agent instance
        agent = cls(algorithm=algorithm, env=env, device=device)

        # Load model
        algo_class = ALGO_CLASS_MAP[algorithm]
        agent.model = algo_class.load(str(model_file), env=agent.vec_env, device=device)

        # Load normalizer
        norm_path = model_dir / "vec_normalize.pkl"
        if norm_path.exists() and isinstance(agent.vec_env, VecNormalize):
            agent.vec_env = VecNormalize.load(str(norm_path), agent.vec_env.venv)

        agent._is_trained = True
        logger.info(f"Model loaded from {model_path}")
        return agent


# ── Agent Factory ─────────────────────────────────────────────


class AgentFactory:
    """Factory for creating trading agents by algorithm name."""

    @staticmethod
    def create(
        algorithm: str | RLAlgorithm,
        env: TradingEnv,
        hyperparams: Optional[dict] = None,
        **kwargs,
    ) -> TradingAgent:
        """Create a trading agent.

        Args:
            algorithm: 'PPO', 'DQN', 'A2C' or RLAlgorithm enum
            env: Trading environment
            hyperparams: Custom hyperparameters
            **kwargs: Additional TradingAgent arguments

        Returns:
            Configured TradingAgent
        """
        if isinstance(algorithm, str):
            algorithm = RLAlgorithm(algorithm.upper())

        return TradingAgent(
            algorithm=algorithm,
            env=env,
            hyperparams=hyperparams,
            **kwargs,
        )

    @staticmethod
    def available_algorithms() -> list[str]:
        """List available algorithm names."""
        return [algo.value for algo in RLAlgorithm]
