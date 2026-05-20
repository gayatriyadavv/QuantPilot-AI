"""
QuantPilot AI — Trading Signal Generator

Generate buy/sell signals from technical indicators with multi-indicator consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class SignalType(IntEnum):
    """Trading signal types."""
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


@dataclass
class Signal:
    """A trading signal with metadata."""
    type: SignalType
    source: str
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: pd.Timestamp
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalGenerator:
    """Generate trading signals from technical indicators.

    Supports individual indicator signals and multi-indicator consensus.
    """

    # ── RSI Signals ───────────────────────────────────────────

    @staticmethod
    def rsi_signals(
        df: pd.DataFrame,
        overbought: float = 70,
        oversold: float = 30,
    ) -> pd.Series:
        """Generate signals from RSI overbought/oversold levels.

        Returns:
            Series of SignalType values
        """
        if "rsi" not in df.columns:
            raise ValueError("RSI column not found. Run TechnicalIndicators first.")

        signals = pd.Series(SignalType.NEUTRAL, index=df.index, dtype=int)
        signals[df["rsi"] < oversold] = SignalType.BUY
        signals[df["rsi"] < 20] = SignalType.STRONG_BUY
        signals[df["rsi"] > overbought] = SignalType.SELL
        signals[df["rsi"] > 80] = SignalType.STRONG_SELL

        return signals

    # ── MACD Signals ──────────────────────────────────────────

    @staticmethod
    def macd_signals(df: pd.DataFrame) -> pd.Series:
        """Generate signals from MACD crossovers."""
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            raise ValueError("MACD columns not found.")

        signals = pd.Series(SignalType.NEUTRAL, index=df.index, dtype=int)

        # MACD crosses above signal line → Buy
        macd_cross_up = (df["macd"] > df["macd_signal"]) & (
            df["macd"].shift(1) <= df["macd_signal"].shift(1)
        )
        # MACD crosses below signal line → Sell
        macd_cross_down = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )

        signals[macd_cross_up] = SignalType.BUY
        signals[macd_cross_down] = SignalType.SELL

        # Strong signals when MACD is also above/below zero
        strong_buy = macd_cross_up & (df["macd"] > 0)
        strong_sell = macd_cross_down & (df["macd"] < 0)
        signals[strong_buy] = SignalType.STRONG_BUY
        signals[strong_sell] = SignalType.STRONG_SELL

        return signals

    # ── Bollinger Band Signals ────────────────────────────────

    @staticmethod
    def bollinger_signals(df: pd.DataFrame) -> pd.Series:
        """Generate signals from Bollinger Band breakouts."""
        if "bb_upper" not in df.columns:
            raise ValueError("Bollinger Band columns not found.")

        signals = pd.Series(SignalType.NEUTRAL, index=df.index, dtype=int)

        # Price touches lower band → Buy (mean reversion)
        signals[df["close"] <= df["bb_lower"]] = SignalType.BUY

        # Price touches upper band → Sell (mean reversion)
        signals[df["close"] >= df["bb_upper"]] = SignalType.SELL

        # Band squeeze (low volatility → breakout incoming)
        if "bb_width" in df.columns:
            avg_width = df["bb_width"].rolling(window=50).mean()
            squeeze = df["bb_width"] < (avg_width * 0.5)
            # During squeeze, amplify the signal
            signals[squeeze & (signals == SignalType.BUY)] = SignalType.STRONG_BUY
            signals[squeeze & (signals == SignalType.SELL)] = SignalType.STRONG_SELL

        return signals

    # ── SMA Crossover Signals ─────────────────────────────────

    @staticmethod
    def sma_crossover_signals(
        df: pd.DataFrame,
        fast_col: str = "sma_20",
        slow_col: str = "sma_50",
    ) -> pd.Series:
        """Generate signals from SMA crossovers (Golden/Death Cross)."""
        if fast_col not in df.columns or slow_col not in df.columns:
            raise ValueError(f"SMA columns not found: {fast_col}, {slow_col}")

        signals = pd.Series(SignalType.NEUTRAL, index=df.index, dtype=int)

        # Golden cross (fast crosses above slow)
        golden = (df[fast_col] > df[slow_col]) & (
            df[fast_col].shift(1) <= df[slow_col].shift(1)
        )
        # Death cross (fast crosses below slow)
        death = (df[fast_col] < df[slow_col]) & (
            df[fast_col].shift(1) >= df[slow_col].shift(1)
        )

        signals[golden] = SignalType.BUY
        signals[death] = SignalType.SELL

        return signals

    # ── Stochastic Signals ────────────────────────────────────

    @staticmethod
    def stochastic_signals(
        df: pd.DataFrame,
        overbought: float = 80,
        oversold: float = 20,
    ) -> pd.Series:
        """Generate signals from Stochastic oscillator."""
        if "stochastic_k" not in df.columns:
            raise ValueError("Stochastic columns not found.")

        signals = pd.Series(SignalType.NEUTRAL, index=df.index, dtype=int)

        # %K crosses above %D in oversold zone → Buy
        if "stochastic_d" in df.columns:
            cross_up = (
                (df["stochastic_k"] > df["stochastic_d"]) &
                (df["stochastic_k"].shift(1) <= df["stochastic_d"].shift(1)) &
                (df["stochastic_k"] < oversold)
            )
            cross_down = (
                (df["stochastic_k"] < df["stochastic_d"]) &
                (df["stochastic_k"].shift(1) >= df["stochastic_d"].shift(1)) &
                (df["stochastic_k"] > overbought)
            )
            signals[cross_up] = SignalType.BUY
            signals[cross_down] = SignalType.SELL
        else:
            signals[df["stochastic_k"] < oversold] = SignalType.BUY
            signals[df["stochastic_k"] > overbought] = SignalType.SELL

        return signals

    # ── Multi-Indicator Consensus ─────────────────────────────

    @staticmethod
    def consensus_signal(
        df: pd.DataFrame,
        weights: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Generate consensus signal from multiple indicators.

        Combines RSI, MACD, Bollinger, SMA crossover, and Stochastic signals
        into a weighted consensus score.

        Args:
            df: DataFrame with all indicators computed
            weights: Dict mapping signal name → weight (default: equal weights)

        Returns:
            DataFrame with added columns: individual signals + consensus
        """
        signal_gen = SignalGenerator()
        result = df.copy()

        # Generate individual signals (with error handling)
        signal_map = {}
        try:
            signal_map["rsi_signal"] = signal_gen.rsi_signals(df)
        except ValueError:
            pass

        try:
            signal_map["macd_signal_ind"] = signal_gen.macd_signals(df)
        except ValueError:
            pass

        try:
            signal_map["bollinger_signal"] = signal_gen.bollinger_signals(df)
        except ValueError:
            pass

        try:
            signal_map["sma_signal"] = signal_gen.sma_crossover_signals(df)
        except ValueError:
            pass

        try:
            signal_map["stochastic_signal"] = signal_gen.stochastic_signals(df)
        except ValueError:
            pass

        if not signal_map:
            logger.warning("No indicators available for consensus signal")
            result["consensus_score"] = 0
            result["consensus_signal"] = SignalType.NEUTRAL
            return result

        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(signal_map) for name in signal_map}

        # Add individual signals to result
        for name, signal_series in signal_map.items():
            result[name] = signal_series

        # Calculate weighted consensus score (-2 to +2)
        consensus = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for name, signal_series in signal_map.items():
            w = weights.get(name, 1.0 / len(signal_map))
            consensus += signal_series.astype(float) * w
            total_weight += w

        if total_weight > 0:
            consensus /= total_weight

        result["consensus_score"] = consensus

        # Convert score to discrete signal
        result["consensus_signal"] = SignalType.NEUTRAL
        result.loc[consensus >= 1.5, "consensus_signal"] = SignalType.STRONG_BUY
        result.loc[(consensus >= 0.5) & (consensus < 1.5), "consensus_signal"] = SignalType.BUY
        result.loc[(consensus <= -0.5) & (consensus > -1.5), "consensus_signal"] = SignalType.SELL
        result.loc[consensus <= -1.5, "consensus_signal"] = SignalType.STRONG_SELL

        # Count agreeing signals
        signal_df = pd.DataFrame(signal_map)
        result["bullish_count"] = (signal_df > 0).sum(axis=1)
        result["bearish_count"] = (signal_df < 0).sum(axis=1)
        result["signal_agreement"] = result[["bullish_count", "bearish_count"]].max(axis=1) / len(signal_map)

        logger.info(f"Consensus signal generated from {len(signal_map)} indicators")
        return result

    # ── Signal Summary ────────────────────────────────────────

    @staticmethod
    def get_latest_signals(df: pd.DataFrame) -> dict:
        """Get the latest signal summary for display.

        Returns dict with current signal state for each indicator.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        summary = {"timestamp": str(df.index[-1]), "price": float(latest["close"])}

        signal_cols = [c for c in df.columns if c.endswith("_signal") or c == "consensus_score"]
        for col in signal_cols:
            if col in latest:
                val = latest[col]
                if isinstance(val, (int, np.integer)):
                    summary[col] = {"value": int(val), "label": SignalType(val).name}
                else:
                    summary[col] = float(val)

        # Add key indicator values
        indicator_snapshot = {}
        for col in ["rsi", "macd", "adx", "atr", "bb_pct", "stochastic_k"]:
            if col in latest and not pd.isna(latest[col]):
                indicator_snapshot[col] = round(float(latest[col]), 4)
        summary["indicators"] = indicator_snapshot

        return summary
