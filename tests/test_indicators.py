"""
QuantPilot AI — Technical Indicator Tests
"""

import numpy as np
import pandas as pd
import pytest

from backend.indicators.technical import IndicatorConfig, IndicatorPipeline, TechnicalIndicators
from backend.indicators.signals import SignalGenerator, SignalType


@pytest.fixture
def ohlcv_data():
    """Create sample OHLCV data with enough history for all indicators."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + abs(np.random.randn(n)) * 2
    low = close - abs(np.random.randn(n)) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestTechnicalIndicators:
    def test_trend_indicators(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_trend_indicators(ohlcv_data)

        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert "adx" in result.columns

    def test_momentum_indicators(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_momentum_indicators(ohlcv_data)

        assert "rsi" in result.columns
        assert "stochastic_k" in result.columns
        assert "stochastic_d" in result.columns
        assert "cci" in result.columns
        assert "williams_r" in result.columns
        assert "roc" in result.columns

    def test_volatility_indicators(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_volatility_indicators(ohlcv_data)

        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "atr" in result.columns
        assert "bb_width" in result.columns

    def test_volume_indicators(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_volume_indicators(ohlcv_data)

        assert "obv" in result.columns
        assert "volume_sma" in result.columns
        assert "volume_ratio" in result.columns

    def test_all_indicators(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_all_indicators(ohlcv_data)

        # Should have original OHLCV + many indicators
        assert len(result.columns) > 20
        # Should have dropped NaN rows
        assert result.isna().sum().sum() == 0

    def test_rsi_range(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_momentum_indicators(ohlcv_data)
        rsi_valid = result["rsi"].dropna()
        assert rsi_valid.min() >= 0
        assert rsi_valid.max() <= 100

    def test_bollinger_band_ordering(self, ohlcv_data):
        calc = TechnicalIndicators()
        result = calc.add_volatility_indicators(ohlcv_data)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_custom_config(self, ohlcv_data):
        config = IndicatorConfig(sma_periods=[10, 30], rsi_period=7, bb_period=15)
        calc = TechnicalIndicators(config)
        result = calc.add_all_indicators(ohlcv_data)

        assert "sma_10" in result.columns
        assert "sma_30" in result.columns
        assert "sma_20" not in result.columns

    def test_get_indicator_names(self):
        calc = TechnicalIndicators()
        names = calc.get_indicator_names()
        assert len(names) > 15
        assert "rsi" in names
        assert "macd" in names


class TestIndicatorPipeline:
    def test_pipeline_single_step(self, ohlcv_data):
        pipeline = IndicatorPipeline()
        pipeline.add_step("trend")
        result = pipeline.run(ohlcv_data)

        assert "sma_20" in result.columns
        assert "rsi" not in result.columns

    def test_pipeline_multiple_steps(self, ohlcv_data):
        pipeline = IndicatorPipeline()
        pipeline.add_step("trend")
        pipeline.add_step("momentum")
        result = pipeline.run(ohlcv_data)

        assert "sma_20" in result.columns
        assert "rsi" in result.columns

    def test_pipeline_all(self, ohlcv_data):
        pipeline = IndicatorPipeline()
        pipeline.add_step("all")
        result = pipeline.run(ohlcv_data)

        assert len(result.columns) > 20

    def test_pipeline_invalid_step(self):
        pipeline = IndicatorPipeline()
        with pytest.raises(ValueError):
            pipeline.add_step("invalid")

    def test_pipeline_chaining(self, ohlcv_data):
        pipeline = IndicatorPipeline()
        result = pipeline.add_step("trend").add_step("momentum")
        assert len(pipeline._steps) == 2


class TestSignalGenerator:
    def test_rsi_signals(self, ohlcv_data):
        calc = TechnicalIndicators()
        df = calc.add_momentum_indicators(ohlcv_data)
        signals = SignalGenerator.rsi_signals(df)

        assert len(signals) == len(df)
        assert set(signals.unique()).issubset({-2, -1, 0, 1, 2})

    def test_macd_signals(self, ohlcv_data):
        calc = TechnicalIndicators()
        df = calc.add_trend_indicators(ohlcv_data)
        signals = SignalGenerator.macd_signals(df)

        assert len(signals) == len(df)

    def test_bollinger_signals(self, ohlcv_data):
        calc = TechnicalIndicators()
        df = calc.add_volatility_indicators(ohlcv_data)
        signals = SignalGenerator.bollinger_signals(df)

        assert len(signals) == len(df)

    def test_consensus_signal(self, ohlcv_data):
        calc = TechnicalIndicators()
        df = calc.add_all_indicators(ohlcv_data)
        result = SignalGenerator.consensus_signal(df)

        assert "consensus_score" in result.columns
        assert "consensus_signal" in result.columns
        assert "bullish_count" in result.columns
        assert "bearish_count" in result.columns

    def test_get_latest_signals(self, ohlcv_data):
        calc = TechnicalIndicators()
        df = calc.add_all_indicators(ohlcv_data)
        df_sig = SignalGenerator.consensus_signal(df)
        latest = SignalGenerator.get_latest_signals(df_sig)

        assert "timestamp" in latest
        assert "price" in latest
        assert "indicators" in latest

    def test_missing_indicator_raises(self, ohlcv_data):
        with pytest.raises(ValueError):
            SignalGenerator.rsi_signals(ohlcv_data)  # No RSI column
