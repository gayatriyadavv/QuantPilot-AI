"""
QuantPilot AI — Data Fetcher Tests
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.data.fetcher import (
    BinanceFetcher,
    DataFetcherFactory,
    YahooFinanceFetcher,
    fetch_multiple_assets,
)
from backend.data.processor import DataProcessor, NormalizationMethod


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(100) * 0.5,
            "high": close + abs(np.random.randn(100)) * 2,
            "low": close - abs(np.random.randn(100)) * 2,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=100).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlcv_with_nans(sample_ohlcv):
    """OHLCV data with missing values."""
    df = sample_ohlcv.copy()
    df.iloc[10:13, df.columns.get_loc("close")] = np.nan
    df.iloc[50, df.columns.get_loc("volume")] = np.nan
    return df


# ── Yahoo Finance Tests ──────────────────────────────────────


class TestYahooFinanceFetcher:
    def test_default_symbols(self):
        fetcher = YahooFinanceFetcher()
        symbols = asyncio.run(fetcher.get_available_symbols())
        assert len(symbols) > 0
        assert "AAPL" in symbols

    def test_normalize_columns(self):
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        result = YahooFinanceFetcher._normalize_columns(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_resample_to_4h(self):
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        df = pd.DataFrame(
            {
                "open": np.random.randn(24),
                "high": np.random.randn(24) + 1,
                "low": np.random.randn(24) - 1,
                "close": np.random.randn(24),
                "volume": np.random.randint(100, 1000, size=24).astype(float),
            },
            index=dates,
        )
        result = YahooFinanceFetcher._resample_to_4h(df)
        assert len(result) == 6  # 24h / 4h = 6


# ── Data Processor Tests ─────────────────────────────────────


class TestDataProcessor:
    def test_handle_missing_ffill(self, sample_ohlcv_with_nans):
        result = DataProcessor.handle_missing_values(sample_ohlcv_with_nans, method="ffill")
        assert result.isna().sum().sum() == 0

    def test_handle_missing_interpolate(self, sample_ohlcv_with_nans):
        result = DataProcessor.handle_missing_values(sample_ohlcv_with_nans, method="interpolate")
        assert result.isna().sum().sum() == 0

    def test_handle_missing_drop(self, sample_ohlcv_with_nans):
        result = DataProcessor.handle_missing_values(sample_ohlcv_with_nans, method="drop")
        assert result.isna().sum().sum() == 0
        assert len(result) < len(sample_ohlcv_with_nans)

    def test_normalize_minmax(self, sample_ohlcv):
        processor = DataProcessor()
        result = processor.normalize(sample_ohlcv, method=NormalizationMethod.MINMAX)
        assert result["close"].min() >= 0
        assert result["close"].max() <= 1

    def test_normalize_zscore(self, sample_ohlcv):
        processor = DataProcessor()
        result = processor.normalize(sample_ohlcv, method=NormalizationMethod.ZSCORE)
        assert abs(result["close"].mean()) < 0.1
        assert abs(result["close"].std() - 1.0) < 0.1

    def test_add_returns(self, sample_ohlcv):
        result = DataProcessor.add_returns(sample_ohlcv)
        assert "return" in result.columns
        assert "log_return" in result.columns
        assert "range" in result.columns
        assert "body" in result.columns

    def test_add_volume_features(self, sample_ohlcv):
        result = DataProcessor.add_volume_features(sample_ohlcv)
        assert "volume_change" in result.columns
        assert "volume_ratio" in result.columns

    def test_train_test_split(self, sample_ohlcv):
        train, val, test = DataProcessor.train_test_split(sample_ohlcv, 0.7, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_walk_forward_split(self, sample_ohlcv):
        splits = DataProcessor.walk_forward_split(sample_ohlcv, train_window=50, test_window=10, step_size=10)
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) == 50
            assert len(test) == 10

    def test_validate_ohlcv_valid(self, sample_ohlcv):
        result = DataProcessor.validate_ohlcv(sample_ohlcv)
        assert result["total_rows"] == 100

    def test_process_pipeline(self, sample_ohlcv):
        processor = DataProcessor()
        result = processor.process_pipeline(sample_ohlcv, add_features=True)
        assert len(result) > 0
        assert "return" in result.columns


# ── Factory Tests ─────────────────────────────────────────────


class TestDataFetcherFactory:
    def test_get_yahoo_fetcher(self):
        fetcher = DataFetcherFactory.get_fetcher("yahoo")
        assert isinstance(fetcher, YahooFinanceFetcher)

    def test_get_binance_fetcher(self):
        fetcher = DataFetcherFactory.get_fetcher("binance")
        assert isinstance(fetcher, BinanceFetcher)

    def test_auto_detect_stock(self):
        fetcher = DataFetcherFactory.auto_detect("AAPL")
        assert isinstance(fetcher, YahooFinanceFetcher)

    def test_auto_detect_crypto(self):
        fetcher = DataFetcherFactory.auto_detect("BTCUSDT")
        assert isinstance(fetcher, BinanceFetcher)

    def test_invalid_source(self):
        with pytest.raises(ValueError):
            DataFetcherFactory.get_fetcher("invalid")
