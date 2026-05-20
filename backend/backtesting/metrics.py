"""
QuantPilot AI — Performance Metrics Calculator

Comprehensive trading performance metrics for backtesting and live evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_returns: Optional[pd.Series] = None

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in days

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0

    # Monthly returns
    monthly_returns: Optional[pd.Series] = None
    yearly_returns: Optional[pd.Series] = None

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, pd.Series):
                result[key] = value.to_dict() if value is not None else None
            elif isinstance(value, (np.floating, np.integer)):
                result[key] = float(value)
            else:
                result[key] = value
        return result

    def summary(self) -> str:
        """Pretty-print performance summary."""
        return f"""
╔══════════════════════════════════════════════════╗
║           PERFORMANCE SUMMARY                    ║
╠══════════════════════════════════════════════════╣
║  Total Return:      {self.total_return:>12.2%}                ║
║  Annual Return:     {self.annualized_return:>12.2%}                ║
║  Sharpe Ratio:      {self.sharpe_ratio:>12.3f}                ║
║  Sortino Ratio:     {self.sortino_ratio:>12.3f}                ║
║  Max Drawdown:      {self.max_drawdown:>12.2%}                ║
║  Calmar Ratio:      {self.calmar_ratio:>12.3f}                ║
╠══════════════════════════════════════════════════╣
║  Total Trades:      {self.total_trades:>12d}                ║
║  Win Rate:          {self.win_rate:>12.2%}                ║
║  Profit Factor:     {self.profit_factor:>12.3f}                ║
║  Expectancy:        {self.expectancy:>12.4f}                ║
║  Avg Win:           {self.avg_win:>12.4f}                ║
║  Avg Loss:          {self.avg_loss:>12.4f}                ║
╠══════════════════════════════════════════════════╣
║  Volatility:        {self.volatility:>12.2%}                ║
║  VaR (95%):         {self.var_95:>12.2%}                ║
║  CVaR (95%):        {self.cvar_95:>12.2%}                ║
╚══════════════════════════════════════════════════╝
"""


class MetricsCalculator:
    """Calculate comprehensive trading performance metrics."""

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.04  # 4% annual risk-free rate

    # ── Core Metrics ──────────────────────────────────────────

    @classmethod
    def calculate_all(
        cls,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: Optional[float] = None,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio values over time
            trades: DataFrame with columns: entry_price, exit_price, pnl, entry_time, exit_time
            benchmark: Series of benchmark values (e.g., buy & hold)
            risk_free_rate: Annual risk-free rate (default: 4%)

        Returns:
            PerformanceMetrics dataclass with all calculated metrics
        """
        rfr = risk_free_rate if risk_free_rate is not None else cls.RISK_FREE_RATE

        metrics = PerformanceMetrics()

        if len(equity_curve) < 2:
            logger.warning("Not enough data for metrics calculation")
            return metrics

        # Daily returns
        returns = equity_curve.pct_change().dropna()

        # ── Return Metrics ──
        metrics.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_days = len(returns)
        n_years = n_days / cls.TRADING_DAYS_PER_YEAR
        if n_years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / n_years) - 1
        metrics.cumulative_returns = (1 + returns).cumprod() - 1

        # ── Volatility ──
        metrics.volatility = returns.std() * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
        downside_returns = returns[returns < 0]
        metrics.downside_volatility = (
            downside_returns.std() * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
            if len(downside_returns) > 0
            else 0
        )

        # ── Sharpe & Sortino ──
        daily_rfr = (1 + rfr) ** (1 / cls.TRADING_DAYS_PER_YEAR) - 1
        excess_returns = returns - daily_rfr

        if returns.std() > 0:
            metrics.sharpe_ratio = (
                excess_returns.mean() / returns.std() * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
            )
        if metrics.downside_volatility > 0:
            metrics.sortino_ratio = (
                (metrics.annualized_return - rfr) / metrics.downside_volatility
            )

        # ── Drawdown ──
        drawdown_series = cls._calculate_drawdown(equity_curve)
        metrics.max_drawdown = drawdown_series.min()
        metrics.avg_drawdown = drawdown_series[drawdown_series < 0].mean() if (drawdown_series < 0).any() else 0

        # Max drawdown duration
        is_in_drawdown = drawdown_series < 0
        if is_in_drawdown.any():
            dd_groups = (~is_in_drawdown).cumsum()
            dd_durations = is_in_drawdown.groupby(dd_groups).sum()
            metrics.max_drawdown_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # ── Calmar ──
        if metrics.max_drawdown < 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

        # ── VaR & CVaR ──
        if len(returns) > 0:
            metrics.var_95 = float(np.percentile(returns, 5))
            metrics.cvar_95 = float(returns[returns <= metrics.var_95].mean()) if (returns <= metrics.var_95).any() else metrics.var_95

        # ── Monthly/Yearly Returns ──
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            monthly = equity_curve.resample("ME").last().pct_change().dropna()
            metrics.monthly_returns = monthly
            yearly = equity_curve.resample("YE").last().pct_change().dropna()
            metrics.yearly_returns = yearly

        # ── Trade Metrics ──
        if trades is not None and len(trades) > 0:
            metrics = cls._calculate_trade_metrics(metrics, trades)

        # ── Benchmark Comparison ──
        if benchmark is not None and len(benchmark) > 1:
            metrics = cls._calculate_benchmark_metrics(metrics, returns, benchmark)

        logger.info(
            f"Metrics calculated: return={metrics.total_return:.2%}, "
            f"sharpe={metrics.sharpe_ratio:.3f}, max_dd={metrics.max_drawdown:.2%}"
        )

        return metrics

    # ── Drawdown Calculation ──────────────────────────────────

    @staticmethod
    def _calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown

    @staticmethod
    def get_drawdown_details(equity_curve: pd.Series) -> pd.DataFrame:
        """Get detailed drawdown periods.

        Returns DataFrame with columns:
            start, trough, end, max_drawdown, duration, recovery
        """
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak

        # Find drawdown periods
        is_dd = drawdown < 0
        dd_starts = is_dd & ~is_dd.shift(1, fill_value=False)
        dd_ends = ~is_dd & is_dd.shift(1, fill_value=False)

        periods = []
        starts = drawdown.index[dd_starts].tolist()
        ends = drawdown.index[dd_ends].tolist()

        # Handle ongoing drawdown
        if len(starts) > len(ends):
            ends.append(drawdown.index[-1])

        for start, end in zip(starts, ends):
            period = drawdown.loc[start:end]
            trough_idx = period.idxmin()
            periods.append({
                "start": start,
                "trough": trough_idx,
                "end": end,
                "max_drawdown": float(period.min()),
                "duration": (end - start).days if hasattr(end - start, "days") else len(period),
            })

        if not periods:
            return pd.DataFrame()

        return pd.DataFrame(periods).sort_values("max_drawdown")

    # ── Trade Metrics ─────────────────────────────────────────

    @classmethod
    def _calculate_trade_metrics(
        cls,
        metrics: PerformanceMetrics,
        trades: pd.DataFrame,
    ) -> PerformanceMetrics:
        """Calculate trade-level metrics."""
        if "pnl" not in trades.columns:
            return metrics

        pnl = trades["pnl"].astype(float)
        metrics.total_trades = len(trades)
        metrics.winning_trades = int((pnl > 0).sum())
        metrics.losing_trades = int((pnl < 0).sum())
        metrics.win_rate = metrics.winning_trades / max(metrics.total_trades, 1)

        # Profit factor
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        metrics.profit_factor = gross_profit / max(gross_loss, 1e-10)

        # Averages
        metrics.avg_win = float(pnl[pnl > 0].mean()) if metrics.winning_trades > 0 else 0
        metrics.avg_loss = float(pnl[pnl < 0].mean()) if metrics.losing_trades > 0 else 0
        metrics.expectancy = float(pnl.mean())
        metrics.largest_win = float(pnl.max())
        metrics.largest_loss = float(pnl.min())

        # Trade duration
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            durations = (
                pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
            )
            valid_durations = durations.dropna()
            if len(valid_durations) > 0:
                metrics.avg_trade_duration = valid_durations.mean().total_seconds() / 86400  # days

        return metrics

    # ── Benchmark Comparison ──────────────────────────────────

    @classmethod
    def _calculate_benchmark_metrics(
        cls,
        metrics: PerformanceMetrics,
        strategy_returns: pd.Series,
        benchmark: pd.Series,
    ) -> PerformanceMetrics:
        """Calculate benchmark-relative metrics."""
        benchmark_returns = benchmark.pct_change().dropna()

        # Align dates
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 2:
            return metrics

        strat = strategy_returns.loc[common_idx]
        bench = benchmark_returns.loc[common_idx]

        metrics.benchmark_return = float((1 + bench).prod() - 1)

        # Beta & Alpha (CAPM)
        if bench.var() > 0:
            cov = np.cov(strat, bench)
            metrics.beta = float(cov[0, 1] / cov[1, 1])
            metrics.alpha = float(
                metrics.annualized_return - cls.RISK_FREE_RATE -
                metrics.beta * (metrics.benchmark_return - cls.RISK_FREE_RATE)
            )

        # Information ratio
        tracking_error = (strat - bench).std() * np.sqrt(cls.TRADING_DAYS_PER_YEAR)
        if tracking_error > 0:
            metrics.information_ratio = float(
                (metrics.annualized_return - metrics.benchmark_return) / tracking_error
            )

        return metrics

    # ── Utility Methods ───────────────────────────────────────

    @staticmethod
    def rolling_sharpe(
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.04,
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        daily_rfr = (1 + risk_free_rate) ** (1 / 252) - 1
        excess = returns - daily_rfr
        rolling_mean = excess.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)

    @staticmethod
    def monthly_return_table(equity_curve: pd.Series) -> pd.DataFrame:
        """Create a monthly returns heatmap-ready table.

        Returns DataFrame with years as rows and months as columns.
        """
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            return pd.DataFrame()

        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            return pd.DataFrame()

        rows = []
        for year in monthly.index.year.unique():
            year_data = monthly[monthly.index.year == year]
            row = {"Year": year}
            for date, val in year_data.items():
                row[date.month] = val
            rows.append(row)

        table = pd.DataFrame(rows).set_index("Year")
        
        # Ensure all 12 months exist
        for m in range(1, 13):
            if m not in table.columns:
                table[m] = np.nan
                
        table = table.sort_index(axis=1)
        table.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        return table
