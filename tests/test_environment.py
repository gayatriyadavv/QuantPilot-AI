"""
QuantPilot AI — Trading Environment Tests
"""

import numpy as np
import pandas as pd
import pytest

from backend.rl.environment import TradingEnv, create_trading_env
from backend.rl.rewards import (
    CompositeReward,
    CostAwareReward,
    DrawdownPenalty,
    LogReturnReward,
    ProfitReward,
    SharpeReward,
    get_default_reward,
)


@pytest.fixture
def trading_data():
    """Create sample data for trading environment."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)  # Ensure positive prices

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n)) * 1.5,
            "low": close - abs(np.random.randn(n)) * 1.5,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=n).astype(float),
            "rsi": np.random.uniform(20, 80, n),
            "macd": np.random.randn(n) * 0.5,
            "atr": np.random.uniform(1, 5, n),
        },
        index=dates,
    )


@pytest.fixture
def env(trading_data):
    """Create a trading environment."""
    return TradingEnv(
        df=trading_data,
        window_size=10,
        initial_balance=100_000,
        commission_rate=0.001,
        slippage=0.0005,
    )


class TestTradingEnv:
    def test_initialization(self, env):
        assert env.initial_balance == 100_000
        assert env.window_size == 10
        assert env.action_space.n == 3

    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert info["cash"] == 100_000
        assert info["shares"] == 0
        assert info["total_value"] == 100_000

    def test_observation_shape(self, env):
        obs, _ = env.reset()
        expected = env.window_size * env.n_features + env.n_portfolio_features
        assert obs.shape == (expected,)
        assert obs.dtype == np.float32

    def test_hold_action(self, env):
        env.reset()
        obs, reward, done, truncated, info = env.step(0)  # Hold
        assert info["shares"] == 0
        assert info["cash"] == 100_000
        assert not done

    def test_buy_action(self, env):
        env.reset()
        obs, reward, done, truncated, info = env.step(1)  # Buy
        assert info["shares"] > 0
        assert info["cash"] < 100_000

    def test_sell_without_position(self, env):
        env.reset()
        obs, reward, done, truncated, info = env.step(2)  # Sell without holding
        assert info["shares"] == 0  # Should be no-op

    def test_buy_then_sell(self, env):
        env.reset()
        env.step(1)  # Buy
        assert env.shares > 0
        obs, reward, done, truncated, info = env.step(2)  # Sell
        assert info["shares"] == 0
        assert info["trade_count"] == 2

    def test_episode_runs_to_completion(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            if steps > 300:
                break
        assert steps > 0

    def test_portfolio_value_tracking(self, env):
        env.reset()
        env.step(1)  # Buy
        value = env.total_value
        assert value > 0
        assert abs(value - 100_000) < 5_000  # Reasonable range

    def test_transaction_costs(self, env):
        env.reset()
        initial = env.cash
        env.step(1)  # Buy
        env.step(2)  # Sell
        # After round trip, cash should be less due to costs
        assert env.cash < initial

    def test_portfolio_return_property(self, env):
        env.reset()
        ret = env.portfolio_return
        assert ret == 0.0  # Initially zero

    def test_is_holding_property(self, env):
        env.reset()
        assert not env.is_holding
        env.step(1)
        assert env.is_holding

    def test_render(self, env, capsys):
        env.reset()
        env.render()
        captured = capsys.readouterr()
        assert "Value" in captured.out

    def test_custom_features(self, trading_data):
        env = TradingEnv(
            df=trading_data,
            window_size=5,
            features=["close", "volume", "rsi"],
        )
        obs, _ = env.reset()
        expected = 5 * 3 + 3  # 5 window * 3 features + 3 portfolio
        assert obs.shape == (expected,)


class TestRewardFunctions:
    def test_profit_reward(self, env):
        reward_fn = ProfitReward(scale=100)
        env.reset()
        env.step(1)
        env.prev_total_value = 100_000
        env.total_value = 101_000
        reward = reward_fn.calculate(env)
        assert reward > 0

    def test_log_return_reward(self, env):
        reward_fn = LogReturnReward(scale=100)
        env.reset()
        env.prev_total_value = 100_000
        env.total_value = 101_000
        reward = reward_fn.calculate(env)
        assert reward > 0

    def test_sharpe_reward(self, env):
        reward_fn = SharpeReward(window=5)
        env.reset()
        for _ in range(10):
            env.prev_total_value = env.total_value
            env.total_value *= 1.001
            reward = reward_fn.calculate(env)
        assert isinstance(reward, float)

    def test_drawdown_penalty(self, env):
        reward_fn = DrawdownPenalty(penalty_factor=2.0, threshold=0.05)
        env.reset()
        env.total_value = 100_000
        reward_fn.calculate(env)  # Set peak
        env.total_value = 90_000  # 10% drawdown
        reward = reward_fn.calculate(env)
        assert reward < 0  # Should be penalized

    def test_cost_aware_reward(self, env):
        reward_fn = CostAwareReward()
        env.reset()
        env.prev_total_value = 100_000
        env.total_value = 100_100
        reward = reward_fn.calculate(env)
        assert isinstance(reward, float)

    def test_composite_reward(self, env):
        reward_fn = CompositeReward([
            (ProfitReward(), 0.5),
            (DrawdownPenalty(), 0.5),
        ])
        env.reset()
        env.prev_total_value = 100_000
        env.total_value = 101_000
        reward = reward_fn.calculate(env)
        assert isinstance(reward, float)

    def test_default_reward_preset(self):
        reward = get_default_reward()
        assert isinstance(reward, CompositeReward)
        assert len(reward.components) == 4

    def test_reward_reset(self):
        reward = get_default_reward()
        reward.reset()  # Should not raise


class TestCreateTradingEnv:
    def test_factory_function(self, trading_data):
        env = create_trading_env(
            df=trading_data,
            window_size=10,
            initial_balance=50_000,
        )
        assert isinstance(env, TradingEnv)
        assert env.initial_balance == 50_000
