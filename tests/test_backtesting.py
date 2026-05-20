"""
QuantPilot AI — Backtesting Engine Tests
"""

import numpy as np
import pandas as pd
import pytest

from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    buy_and_hold_strategy,
    sma_crossover_strategy,
)
from backend.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from backend.backtesting.report import ReportGenerator


@pytest.fixture
def ohlcv_data():
    """Create sample OHLCV data with SMA columns."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)

    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n)) * 1.5,
            "low": close - abs(np.random.randn(n)) * 1.5,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=n).astype(float),
        },
        index=dates,
    )

    # Add SMA indicators for sma_crossover_strategy
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df = df.dropna()

    return df


@pytest.fixture
def equity_curve():
    """Create sample equity curve."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    returns = np.random.randn(n) * 0.01 + 0.0003
    equity = 100_000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


class TestBacktestEngine:
    def test_buy_and_hold(self, ohlcv_data):
        config = BacktestConfig(initial_balance=100_000)
        engine = BacktestEngine(config)
        result = engine.run(ohlcv_data, buy_and_hold_strategy, "TEST")

        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert len(result.equity_curve) == len(ohlcv_data)
        assert result.metrics.total_trades >= 1

    def test_sma_crossover(self, ohlcv_data):
        config = BacktestConfig(initial_balance=100_000)
        engine = BacktestEngine(config)
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_commission_impact(self, ohlcv_data):
        # No commission
        config_free = BacktestConfig(initial_balance=100_000, commission_rate=0)
        engine_free = BacktestEngine(config_free)
        result_free = engine_free.run(ohlcv_data, sma_crossover_strategy, "TEST")

        # High commission
        config_high = BacktestConfig(initial_balance=100_000, commission_rate=0.01)
        engine_high = BacktestEngine(config_high)
        result_high = engine_high.run(ohlcv_data, sma_crossover_strategy, "TEST")

        # Higher commission should result in lower final value
        if result_free.metrics.total_trades > 0 and result_high.metrics.total_trades > 0:
            assert result_free.equity_curve.iloc[-1] >= result_high.equity_curve.iloc[-1]

    def test_stop_loss(self, ohlcv_data):
        config = BacktestConfig(
            initial_balance=100_000,
            stop_loss=0.03,
        )
        engine = BacktestEngine(config)
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        assert isinstance(result, BacktestResult)

    def test_take_profit(self, ohlcv_data):
        config = BacktestConfig(
            initial_balance=100_000,
            take_profit=0.05,
        )
        engine = BacktestEngine(config)
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        assert isinstance(result, BacktestResult)

    def test_result_to_dict(self, ohlcv_data):
        engine = BacktestEngine()
        result = engine.run(ohlcv_data, buy_and_hold_strategy, "TEST")
        d = result.to_dict()

        assert "symbol" in d
        assert "metrics" in d
        assert "equity_curve" in d
        assert d["symbol"] == "TEST"

    def test_custom_strategy(self, ohlcv_data):
        """Test with a custom strategy function."""
        def always_hold(df, step, portfolio):
            return 0  # Always hold

        engine = BacktestEngine()
        result = engine.run(ohlcv_data, always_hold, "TEST")

        assert result.metrics.total_trades == 0
        assert result.equity_curve.iloc[-1] == 100_000  # No change

    def test_benchmark_comparison(self, ohlcv_data):
        engine = BacktestEngine()
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST", benchmark_col="close")

        assert result.benchmark is not None
        assert len(result.benchmark) == len(ohlcv_data)


class TestMetricsCalculator:
    def test_calculate_all(self, equity_curve):
        metrics = MetricsCalculator.calculate_all(equity_curve)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return != 0
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown <= 0
        assert metrics.volatility > 0

    def test_empty_equity(self):
        equity = pd.Series([100_000])
        metrics = MetricsCalculator.calculate_all(equity)
        assert metrics.total_return == 0

    def test_drawdown_details(self, equity_curve):
        details = MetricsCalculator.get_drawdown_details(equity_curve)
        if not details.empty:
            assert "start" in details.columns
            assert "max_drawdown" in details.columns
            assert (details["max_drawdown"] <= 0).all()

    def test_rolling_sharpe(self, equity_curve):
        returns = equity_curve.pct_change().dropna()
        rolling = MetricsCalculator.rolling_sharpe(returns, window=30)

        assert len(rolling) == len(returns)
        valid = rolling.dropna()
        assert len(valid) > 0

    def test_monthly_return_table(self, equity_curve):
        table = MetricsCalculator.monthly_return_table(equity_curve)

        if not table.empty:
            assert table.shape[1] <= 12  # At most 12 months

    def test_with_trades(self, equity_curve):
        trades = pd.DataFrame({
            "entry_price": [100, 110, 105],
            "exit_price": [110, 105, 115],
            "pnl": [10, -5, 10],
            "entry_time": pd.date_range("2024-01-01", periods=3, freq="30D"),
            "exit_time": pd.date_range("2024-01-15", periods=3, freq="30D"),
        })

        metrics = MetricsCalculator.calculate_all(equity_curve, trades=trades)
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate > 0.5

    def test_with_benchmark(self, equity_curve):
        benchmark = equity_curve * 0.95  # Slightly underperforming
        metrics = MetricsCalculator.calculate_all(equity_curve, benchmark=benchmark)

        assert metrics.benchmark_return != 0
        assert metrics.beta != 0

    def test_metrics_to_dict(self, equity_curve):
        metrics = MetricsCalculator.calculate_all(equity_curve)
        d = metrics.to_dict()

        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert isinstance(d["total_return"], float)

    def test_metrics_summary(self, equity_curve):
        metrics = MetricsCalculator.calculate_all(equity_curve)
        summary = metrics.summary()

        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary


class TestReportGenerator:
    def test_json_report(self, ohlcv_data, tmp_path):
        engine = BacktestEngine()
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        report = ReportGenerator.generate_json_report(result)
        assert "metadata" in report
        assert "performance" in report
        assert "trades" in report

    def test_json_report_save(self, ohlcv_data, tmp_path):
        engine = BacktestEngine()
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        filepath = str(tmp_path / "report.json")
        ReportGenerator.generate_json_report(result, filepath=filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert data["metadata"]["symbol"] == "TEST"

    def test_html_report(self, ohlcv_data, tmp_path):
        engine = BacktestEngine()
        result = engine.run(ohlcv_data, sma_crossover_strategy, "TEST")

        filepath = str(tmp_path / "report.html")
        html = ReportGenerator.generate_html_report(result, filepath=filepath)

        assert "QuantPilot AI" in html
        assert "TEST" in html

    def test_compare_backtests(self, ohlcv_data):
        engine = BacktestEngine()

        result1 = engine.run(ohlcv_data, sma_crossover_strategy, "SMA")
        engine2 = BacktestEngine()
        result2 = engine2.run(ohlcv_data, buy_and_hold_strategy, "B&H")

        comparison = ReportGenerator.compare_backtests(
            [result1, result2], ["SMA Crossover", "Buy & Hold"]
        )

        assert len(comparison.columns) == 2
        assert "SMA Crossover" in comparison.columns
        assert "Buy & Hold" in comparison.columns
