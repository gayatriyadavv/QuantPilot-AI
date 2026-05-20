"""
QuantPilot AI — Data Preprocessing Pipeline

Handles missing values, normalization, feature engineering, and data quality checks.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NormalizationMethod(str, Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    LOG_RETURN = "log_return"
    PCT_CHANGE = "pct_change"
    NONE = "none"


class DataProcessor:
    """Comprehensive data preprocessing pipeline for financial time series."""

    def __init__(self):
        self._scalers: dict[str, MinMaxScaler | StandardScaler] = {}

    # ── Missing Data Handling ─────────────────────────────────

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        method: str = "ffill",
        max_gap: int = 5,
    ) -> pd.DataFrame:
        """Handle missing values in OHLCV data.

        Args:
            df: Input DataFrame
            method: 'ffill' (forward fill), 'interpolate', or 'drop'
            max_gap: Maximum consecutive NaN gap to fill

        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)
        nan_count = df.isna().sum().sum()

        if nan_count == 0:
            return df

        logger.info(f"Handling {nan_count} missing values (method={method})")

        df = df.copy()

        if method == "ffill":
            df = df.ffill(limit=max_gap)
            df = df.bfill(limit=max_gap)  # Backfill remaining at start
        elif method == "interpolate":
            df = df.interpolate(method="time", limit=max_gap)
            df = df.bfill(limit=max_gap)
        elif method == "drop":
            df = df.dropna()

        # Drop any remaining NaN rows
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            df = df.dropna()
            logger.warning(f"Dropped {remaining_nans} remaining NaN values")

        logger.info(f"Cleaned data: {original_len} → {len(df)} rows")
        return df

    # ── Normalization ─────────────────────────────────────────

    def normalize(
        self,
        df: pd.DataFrame,
        method: NormalizationMethod = NormalizationMethod.MINMAX,
        columns: Optional[list[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Normalize DataFrame columns.

        Args:
            df: Input DataFrame
            method: Normalization method
            columns: Columns to normalize (None = all numeric)
            fit: Whether to fit the scaler (False for transform-only)

        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == NormalizationMethod.NONE:
            return df

        if method == NormalizationMethod.MINMAX:
            scaler_key = "minmax"
            if fit or scaler_key not in self._scalers:
                scaler = MinMaxScaler()
                df[columns] = scaler.fit_transform(df[columns])
                self._scalers[scaler_key] = scaler
            else:
                df[columns] = self._scalers[scaler_key].transform(df[columns])

        elif method == NormalizationMethod.ZSCORE:
            scaler_key = "zscore"
            if fit or scaler_key not in self._scalers:
                scaler = StandardScaler()
                df[columns] = scaler.fit_transform(df[columns])
                self._scalers[scaler_key] = scaler
            else:
                df[columns] = self._scalers[scaler_key].transform(df[columns])

        elif method == NormalizationMethod.LOG_RETURN:
            for col in columns:
                df[col] = np.log(df[col] / df[col].shift(1))
            df = df.iloc[1:]  # Drop first row (NaN from shift)

        elif method == NormalizationMethod.PCT_CHANGE:
            for col in columns:
                df[col] = df[col].pct_change()
            df = df.iloc[1:]

        return df

    def inverse_normalize(
        self,
        data: np.ndarray,
        method: NormalizationMethod = NormalizationMethod.MINMAX,
    ) -> np.ndarray:
        """Inverse transform normalized data back to original scale."""
        scaler_key = method.value
        if scaler_key not in self._scalers:
            raise ValueError(f"No fitted scaler found for method: {method}")
        return self._scalers[scaler_key].inverse_transform(data)

    # ── Feature Engineering ───────────────────────────────────

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features to OHLCV data."""
        df = df.copy()

        # Simple returns
        df["return"] = df["close"].pct_change()

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Intraday range
        df["range"] = (df["high"] - df["low"]) / df["close"]

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Body size (close vs open)
        df["body"] = (df["close"] - df["open"]) / df["open"]

        # Upper/lower wicks
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        # Volume change
        df["volume_change"] = df["volume"].pct_change()

        # Volume ratio (vs 20-day average)
        vol_sma = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / vol_sma

        # Dollar volume
        df["dollar_volume"] = df["close"] * df["volume"]

        return df

    @staticmethod
    def add_rolling_features(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
        """Add rolling statistical features."""
        if windows is None:
            windows = [5, 10, 20, 50]

        df = df.copy()

        for w in windows:
            # Rolling returns
            df[f"return_{w}d"] = df["close"].pct_change(periods=w)

            # Rolling volatility
            df[f"volatility_{w}d"] = df["close"].pct_change().rolling(window=w).std()

            # Rolling min/max (for drawdown)
            df[f"high_{w}d"] = df["high"].rolling(window=w).max()
            df[f"low_{w}d"] = df["low"].rolling(window=w).min()

        return df

    # ── Data Splitting ────────────────────────────────────────

    @staticmethod
    def train_test_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data chronologically.

        Args:
            df: Input DataFrame (must be sorted by date)
            train_ratio: Fraction for training
            validation_ratio: Fraction for validation (rest is test)

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    @staticmethod
    def walk_forward_split(
        df: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
        step_size: int = 21,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Walk-forward analysis splits.

        Args:
            df: Input DataFrame
            train_window: Training window size (days)
            test_window: Test window size (days)
            step_size: Step between windows (days)

        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        n = len(df)
        start = 0

        while start + train_window + test_window <= n:
            train_end = start + train_window
            test_end = train_end + test_window

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            splits.append((train_df, test_df))

            start += step_size

        logger.info(f"Walk-forward: {len(splits)} splits (train={train_window}, test={test_window})")
        return splits

    # ── Data Validation ───────────────────────────────────────

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> dict:
        """Validate OHLCV data quality.

        Returns:
            Dictionary with validation results and issues found.
        """
        issues = []
        stats = {
            "total_rows": len(df),
            "date_range": f"{df.index[0]} → {df.index[-1]}" if len(df) > 0 else "empty",
            "nan_counts": df.isna().sum().to_dict(),
        }

        if df.empty:
            issues.append("DataFrame is empty")
            return {"valid": False, "issues": issues, "stats": stats}

        # Check OHLC consistency
        invalid_ohlc = df[
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        ]
        if len(invalid_ohlc) > 0:
            issues.append(f"{len(invalid_ohlc)} rows with invalid OHLC relationships")

        # Check for negative values
        neg_prices = df[(df[["open", "high", "low", "close"]] < 0).any(axis=1)]
        if len(neg_prices) > 0:
            issues.append(f"{len(neg_prices)} rows with negative prices")

        # Check for zero volume
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            issues.append(f"{zero_vol} rows with zero volume")

        # Check for duplicates
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate datetime entries")

        # Check sorting
        if not df.index.is_monotonic_increasing:
            issues.append("Data is not sorted chronologically")

        # Large gaps
        if len(df) > 1:
            time_diffs = pd.Series(df.index).diff()
            median_diff = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_diff * 5].dropna()
            if len(large_gaps) > 0:
                issues.append(f"{len(large_gaps)} suspicious time gaps detected")

        stats["issues"] = issues
        stats["valid"] = len(issues) == 0

        if issues:
            for issue in issues:
                logger.warning(f"Data validation: {issue}")
        else:
            logger.info("Data validation passed ✅")

        return stats

    # ── Full Pipeline ─────────────────────────────────────────

    def process_pipeline(
        self,
        df: pd.DataFrame,
        add_features: bool = True,
        normalize_method: NormalizationMethod = NormalizationMethod.NONE,
        handle_missing: str = "ffill",
    ) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Args:
            df: Raw OHLCV DataFrame
            add_features: Whether to add engineered features
            normalize_method: Normalization method to apply
            handle_missing: Missing value strategy

        Returns:
            Processed DataFrame ready for model consumption.
        """
        logger.info(f"Running preprocessing pipeline on {len(df)} rows")

        # 1. Validate
        self.validate_ohlcv(df)

        # 2. Handle missing values
        df = self.handle_missing_values(df, method=handle_missing)

        # 3. Sort chronologically
        df = df.sort_index()

        # 4. Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # 5. Feature engineering
        if add_features:
            df = self.add_returns(df)
            df = self.add_volume_features(df)
            df = self.add_rolling_features(df)

        # 6. Drop NaN rows from feature engineering
        df = df.dropna()

        # 7. Normalize
        if normalize_method != NormalizationMethod.NONE:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = self.normalize(df, method=normalize_method, columns=numeric_cols)

        logger.info(f"Pipeline complete: {len(df)} rows, {len(df.columns)} features")
        return df
