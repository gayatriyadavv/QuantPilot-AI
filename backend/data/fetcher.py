"""
QuantPilot AI — Market Data Fetcher

Unified interface for fetching OHLCV data from Yahoo Finance and Binance.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import Timeframe, get_settings


# ── Abstract Base ─────────────────────────────────────────────


class BaseDataFetcher(ABC):
    """Abstract base class for market data fetchers."""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTCUSDT')
            timeframe: Candle interval ('1m', '5m', '1h', '1d', etc.)
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            period: Alternative to start/end (e.g., '1y', '2y', 'max')

        Returns:
            DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: DatetimeIndex
        """
        ...

    @abstractmethod
    async def get_available_symbols(self) -> list[str]:
        """Return list of available symbols from this data source."""
        ...


# ── Yahoo Finance Fetcher ────────────────────────────────────


class YahooFinanceFetcher(BaseDataFetcher):
    """Fetch equity and ETF data from Yahoo Finance via yfinance."""

    # yfinance interval mapping
    _INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "1h",  # 4h not directly supported, we'll resample
        "1d": "1d",
        "1wk": "1wk",
    }

    # Default popular symbols
    DEFAULT_SYMBOLS = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM",
        "V", "JNJ", "WMT", "PG", "UNH", "HD", "BAC", "SPY", "QQQ",
    ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        logger.info(f"Fetching {symbol} data from Yahoo Finance (timeframe={timeframe})")

        interval = self._INTERVAL_MAP.get(timeframe, "1d")

        # Run yfinance in executor to avoid blocking
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            None,
            lambda: self._download_sync(symbol, interval, start_date, end_date, period),
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        # Normalize column names
        df = self._normalize_columns(df)

        # Handle 4h by resampling from 1h
        if timeframe == "4h" and interval == "1h":
            df = self._resample_to_4h(df)

        logger.info(f"Fetched {len(df)} rows for {symbol} ({timeframe})")
        return df

    def _download_sync(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: Optional[str],
    ) -> pd.DataFrame:
        """Synchronous download wrapper."""
        ticker = yf.Ticker(symbol)

        kwargs = {"interval": interval}
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start_date or (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            kwargs["end"] = end_date or datetime.now().strftime("%Y-%m-%d")

        try:
            df = ticker.history(**kwargs)
            return df
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Keep only OHLCV columns
        keep_cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.index.name = "datetime"
        return df

    @staticmethod
    def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1h data to 4h candles."""
        return df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    async def get_available_symbols(self) -> list[str]:
        return self.DEFAULT_SYMBOLS

    async def get_ticker_info(self, symbol: str) -> dict:
        """Get detailed ticker information."""
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(
            None, lambda: yf.Ticker(symbol).info
        )
        return info or {}


# ── Binance Fetcher ───────────────────────────────────────────


class BinanceFetcher(BaseDataFetcher):
    """Fetch crypto data from Binance API."""

    _INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1wk": "1w",
    }

    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT",
    ]

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-load Binance client."""
        if self._client is None:
            settings = get_settings()
            if not settings.has_binance:
                raise ValueError(
                    "Binance API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET."
                )
            from binance.client import Client
            self._client = Client(settings.binance_api_key, settings.binance_api_secret)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        logger.info(f"Fetching {symbol} data from Binance (timeframe={timeframe})")

        interval = self._INTERVAL_MAP.get(timeframe, "1d")

        # Convert period to start_date if needed
        if period and not start_date:
            period_map = {
                "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
            }
            days = period_map.get(period, 365)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        start_str = f"{start_date}" if start_date else "1 Jan, 2020"

        loop = asyncio.get_running_loop()
        klines = await loop.run_in_executor(
            None,
            lambda: self._get_client().get_historical_klines(
                symbol, interval, start_str, end_date
            ),
        )

        if not klines:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Parse klines into DataFrame
        df = pd.DataFrame(klines, columns=[
            "datetime", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])

        # Convert types
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "datetime"

        logger.info(f"Fetched {len(df)} rows for {symbol} ({timeframe})")
        return df

    async def get_available_symbols(self) -> list[str]:
        return self.DEFAULT_SYMBOLS


# ── Factory ───────────────────────────────────────────────────


class DataFetcherFactory:
    """Factory for creating the appropriate data fetcher."""

    _fetchers: dict[str, BaseDataFetcher] = {}

    @classmethod
    def get_fetcher(cls, source: str = "yahoo") -> BaseDataFetcher:
        """Get a data fetcher by source name.

        Args:
            source: 'yahoo' or 'binance'

        Returns:
            Configured data fetcher instance.
        """
        source = source.lower()

        if source not in cls._fetchers:
            if source == "yahoo":
                cls._fetchers[source] = YahooFinanceFetcher()
            elif source == "binance":
                cls._fetchers[source] = BinanceFetcher()
            else:
                raise ValueError(f"Unknown data source: {source}. Use 'yahoo' or 'binance'.")

        return cls._fetchers[source]

    @classmethod
    def auto_detect(cls, symbol: str) -> BaseDataFetcher:
        """Auto-detect the appropriate fetcher based on symbol format.

        Binance symbols typically end in 'USDT', 'BTC', 'ETH', etc.
        """
        crypto_suffixes = ("USDT", "BTC", "ETH", "BNB", "BUSD")
        if symbol.upper().endswith(crypto_suffixes) and not any(c in symbol for c in ["-", ".", "^"]):
            return cls.get_fetcher("binance")
        return cls.get_fetcher("yahoo")


# ── Multi-Asset Fetcher ───────────────────────────────────────


async def fetch_multiple_assets(
    symbols: list[str],
    timeframe: str = "1d",
    period: str = "2y",
    source: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple assets concurrently.

    Args:
        symbols: List of ticker symbols
        timeframe: Candle interval
        period: Historical period
        source: 'yahoo', 'binance', or 'auto'

    Returns:
        Dictionary mapping symbol → DataFrame
    """
    results: dict[str, pd.DataFrame] = {}

    async def _fetch_one(sym: str):
        try:
            if source == "auto":
                fetcher = DataFetcherFactory.auto_detect(sym)
            else:
                fetcher = DataFetcherFactory.get_fetcher(source)
            df = await fetcher.fetch_ohlcv(sym, timeframe=timeframe, period=period)
            results[sym] = df
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")
            results[sym] = pd.DataFrame()

    # Fetch with controlled concurrency
    tasks = [_fetch_one(sym) for sym in symbols]
    await asyncio.gather(*tasks)

    return results


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument("--symbols", type=str, default="AAPL,GOOGL,MSFT", help="Comma-separated symbols")
    parser.add_argument("--period", type=str, default="2y", help="Historical period")
    parser.add_argument("--timeframe", type=str, default="1d", help="Candle interval")
    parser.add_argument("--source", type=str, default="auto", help="Data source: yahoo, binance, auto")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    async def main():
        start = time.time()
        data = await fetch_multiple_assets(symbols, args.timeframe, args.period, args.source)
        elapsed = time.time() - start

        for sym, df in data.items():
            if not df.empty:
                print(f"\n{'=' * 50}")
                print(f"  {sym} — {len(df)} rows ({args.timeframe})")
                print(f"  Date range: {df.index[0]} → {df.index[-1]}")
                print(f"  Close: ${df['close'].iloc[-1]:.2f}")
                print(f"{'=' * 50}")
            else:
                print(f"\n  ⚠ {sym} — No data")

        print(f"\n✅ Fetched {len(symbols)} assets in {elapsed:.1f}s")

    asyncio.run(main())
