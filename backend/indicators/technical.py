"""
QuantPilot AI — Technical Indicator Calculator

Comprehensive technical indicator computation using the `ta` library.
Supports trend, momentum, volatility, and volume indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ta
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
    from ta.trend import (
        SMAIndicator,
        EMAIndicator,
        MACD,
        ADXIndicator,
        CCIIndicator,
    )
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("'ta' library not installed. Using manual calculations.")


# ── Configuration ─────────────────────────────────────────────


@dataclass
class IndicatorConfig:
    """Configuration for technical indicator parameters."""

    # Trend
    sma_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = field(default_factory=lambda: [12, 26])

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Momentum
    rsi_period: int = 14
    stochastic_period: int = 14
    stochastic_smooth: int = 3
    cci_period: int = 20
    williams_period: int = 14
    roc_period: int = 12

    # Volatility
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    keltner_period: int = 20

    # Volume
    volume_sma_period: int = 20

    # ADX
    adx_period: int = 14


# ── Indicator Calculator ─────────────────────────────────────


class TechnicalIndicators:
    """Calculate technical indicators on OHLCV data.

    Usage:
        calc = TechnicalIndicators()
        df_with_indicators = calc.add_all_indicators(ohlcv_df)

        # Or selectively:
        df = calc.add_trend_indicators(ohlcv_df)
        df = calc.add_momentum_indicators(df)
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()

    # ── Trend Indicators ──────────────────────────────────────

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators: SMA, EMA, MACD, ADX."""
        df = df.copy()

        # Simple Moving Averages
        for period in self.config.sma_periods:
            col_name = f"sma_{period}"
            if TA_AVAILABLE:
                indicator = SMAIndicator(close=df["close"], window=period)
                df[col_name] = indicator.sma_indicator()
            else:
                df[col_name] = df["close"].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in self.config.ema_periods:
            col_name = f"ema_{period}"
            if TA_AVAILABLE:
                indicator = EMAIndicator(close=df["close"], window=period)
                df[col_name] = indicator.ema_indicator()
            else:
                df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        if TA_AVAILABLE:
            macd = MACD(
                close=df["close"],
                window_slow=self.config.macd_slow,
                window_fast=self.config.macd_fast,
                window_sign=self.config.macd_signal,
            )
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_histogram"] = macd.macd_diff()
        else:
            ema_fast = df["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ADX (Average Directional Index)
        if TA_AVAILABLE:
            adx = ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.adx_period,
            )
            df["adx"] = adx.adx()
        else:
            df["adx"] = self._manual_adx(df)

        logger.debug("Added trend indicators: SMA, EMA, MACD, ADX")
        return df

    # ── Momentum Indicators ───────────────────────────────────

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC."""
        df = df.copy()

        # RSI
        if TA_AVAILABLE:
            rsi = RSIIndicator(close=df["close"], window=self.config.rsi_period)
            df["rsi"] = rsi.rsi()
        else:
            df["rsi"] = self._manual_rsi(df["close"], self.config.rsi_period)

        # Stochastic Oscillator
        if TA_AVAILABLE:
            stoch = StochasticOscillator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.stochastic_period,
                smooth_window=self.config.stochastic_smooth,
            )
            df["stochastic_k"] = stoch.stoch()
            df["stochastic_d"] = stoch.stoch_signal()
        else:
            lowest_low = df["low"].rolling(window=self.config.stochastic_period).min()
            highest_high = df["high"].rolling(window=self.config.stochastic_period).max()
            df["stochastic_k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
            df["stochastic_d"] = df["stochastic_k"].rolling(window=self.config.stochastic_smooth).mean()

        # CCI (Commodity Channel Index)
        if TA_AVAILABLE:
            cci = CCIIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.cci_period,
            )
            df["cci"] = cci.cci()
        else:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = typical_price.rolling(window=self.config.cci_period).mean()
            mad = typical_price.rolling(window=self.config.cci_period).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            df["cci"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        # Williams %R
        if TA_AVAILABLE:
            williams = WilliamsRIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                lbp=self.config.williams_period,
            )
            df["williams_r"] = williams.williams_r()
        else:
            highest_high = df["high"].rolling(window=self.config.williams_period).max()
            lowest_low = df["low"].rolling(window=self.config.williams_period).min()
            df["williams_r"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-10)

        # Rate of Change
        if TA_AVAILABLE:
            roc = ROCIndicator(close=df["close"], window=self.config.roc_period)
            df["roc"] = roc.roc()
        else:
            df["roc"] = df["close"].pct_change(periods=self.config.roc_period) * 100

        logger.debug("Added momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC")
        return df

    # ── Volatility Indicators ─────────────────────────────────

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators: Bollinger Bands, ATR, Keltner Channels."""
        df = df.copy()

        # Bollinger Bands
        if TA_AVAILABLE:
            bb = BollingerBands(
                close=df["close"],
                window=self.config.bb_period,
                window_dev=self.config.bb_std,
            )
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = bb.bollinger_wband()
            df["bb_pct"] = bb.bollinger_pband()
        else:
            sma = df["close"].rolling(window=self.config.bb_period).mean()
            std = df["close"].rolling(window=self.config.bb_period).std()
            df["bb_upper"] = sma + (self.config.bb_std * std)
            df["bb_middle"] = sma
            df["bb_lower"] = sma - (self.config.bb_std * std)
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ATR (Average True Range)
        if TA_AVAILABLE:
            atr = AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.atr_period,
            )
            df["atr"] = atr.average_true_range()
        else:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = true_range.rolling(window=self.config.atr_period).mean()

        # ATR percentage (normalized)
        df["atr_pct"] = df["atr"] / df["close"] * 100

        logger.debug("Added volatility indicators: Bollinger Bands, ATR")
        return df

    # ── Volume Indicators ─────────────────────────────────────

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators: OBV, Volume SMA, VWAP."""
        df = df.copy()

        # On-Balance Volume
        if TA_AVAILABLE:
            obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
            df["obv"] = obv.on_balance_volume()
        else:
            obv_values = [0]
            for i in range(1, len(df)):
                if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                    obv_values.append(obv_values[-1] + df["volume"].iloc[i])
                elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                    obv_values.append(obv_values[-1] - df["volume"].iloc[i])
                else:
                    obv_values.append(obv_values[-1])
            df["obv"] = obv_values

        # Volume SMA
        df["volume_sma"] = df["volume"].rolling(window=self.config.volume_sma_period).mean()

        # Volume ratio
        df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

        logger.debug("Added volume indicators: OBV, Volume SMA, Volume Ratio")
        return df

    # ── All Indicators ────────────────────────────────────────

    def add_all_indicators(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Add all technical indicators to OHLCV data.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            drop_na: Whether to drop NaN rows from lookback periods

        Returns:
            DataFrame with all indicator columns added
        """
        logger.info(f"Computing all technical indicators for {len(df)} rows")

        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)

        if drop_na:
            original_len = len(df)
            df = df.dropna()
            dropped = original_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows (indicator lookback NaN)")

        indicator_cols = [
            c for c in df.columns
            if c not in ["open", "high", "low", "close", "volume"]
        ]
        logger.info(
            f"Added {len(indicator_cols)} indicator columns: "
            f"{', '.join(indicator_cols[:8])}{'...' if len(indicator_cols) > 8 else ''}"
        )

        return df

    def get_indicator_names(self) -> list[str]:
        """Return list of all indicator column names."""
        return [
            # Trend
            *[f"sma_{p}" for p in self.config.sma_periods],
            *[f"ema_{p}" for p in self.config.ema_periods],
            "macd", "macd_signal", "macd_histogram", "adx",
            # Momentum
            "rsi", "stochastic_k", "stochastic_d", "cci", "williams_r", "roc",
            # Volatility
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct", "atr", "atr_pct",
            # Volume
            "obv", "volume_sma", "volume_ratio",
        ]

    # ── Manual Fallback Calculations ──────────────────────────

    @staticmethod
    def _manual_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _manual_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX manually (simplified)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx


# ── Pipeline Helper ───────────────────────────────────────────


class IndicatorPipeline:
    """Chain multiple indicator operations for custom pipelines.

    Usage:
        pipeline = IndicatorPipeline()
        pipeline.add_step("trend")
        pipeline.add_step("momentum")
        df = pipeline.run(ohlcv_df)
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.calculator = TechnicalIndicators(config)
        self._steps: list[str] = []

    def add_step(self, indicator_group: str) -> "IndicatorPipeline":
        """Add an indicator group to the pipeline.

        Args:
            indicator_group: 'trend', 'momentum', 'volatility', 'volume', or 'all'
        """
        valid = {"trend", "momentum", "volatility", "volume", "all"}
        if indicator_group not in valid:
            raise ValueError(f"Invalid group: {indicator_group}. Must be one of {valid}")
        self._steps.append(indicator_group)
        return self

    def run(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Execute the pipeline."""
        group_map = {
            "trend": self.calculator.add_trend_indicators,
            "momentum": self.calculator.add_momentum_indicators,
            "volatility": self.calculator.add_volatility_indicators,
            "volume": self.calculator.add_volume_indicators,
            "all": self.calculator.add_all_indicators,
        }

        for step in self._steps:
            if step == "all":
                return self.calculator.add_all_indicators(df, drop_na=drop_na)
            df = group_map[step](df)

        if drop_na:
            df = df.dropna()

        return df
