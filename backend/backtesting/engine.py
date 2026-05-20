"""
QuantPilot AI — Backtesting Engine

Run trading strategies on historical data with full performance evaluation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from backend.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from backend.config import get_settings
from backend.data.fetcher import DataFetcherFactory
from backend.indicators.technical import TechnicalIndicators


@dataclass
class TradeRecord:
    """Single trade record for backtesting."""

    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    entry_time: Any = None
    exit_time: Any = None
    strategy: str = "rl_agent"

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestConfig:
    """Backtesting configuration parameters."""

    initial_balance: float = 100_000.0
    commission_rate: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.95  # % of capital per trade
    allow_short: bool = False
    max_positions: int = 1
    stop_loss: Optional[float] = None  # e.g., 0.05 = 5%
    take_profit: Optional[float] = None  # e.g., 0.10 = 10%


class BacktestEngine:
    """Event-driven backtesting engine.

    Runs a trading strategy (RL agent or rule-based) on historical data,
    applying realistic transaction costs, slippage, and risk rules.

    Usage:
        engine = BacktestEngine(config=BacktestConfig())
        results = engine.run(
            data=ohlcv_df_with_indicators,
            strategy=my_strategy_fn,
            symbol="AAPL",
        )
        print(results.metrics.summary())
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._reset()

    def _reset(self):
        """Reset engine state for a new backtest."""
        self.cash = self.config.initial_balance
        self.position_qty = 0.0
        self.position_price = 0.0
        self.position_side = None  # LONG or SHORT
        self.equity_curve: list[float] = []
        self.equity_dates: list = []
        self.trades: list[TradeRecord] = []
        self.actions_log: list[dict] = []

    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable[[pd.DataFrame, int, dict], int],
        symbol: str = "UNKNOWN",
        benchmark_col: str = "close",
    ) -> BacktestResult:
        """Run a full backtest.

        Args:
            data: OHLCV DataFrame (ideally with indicators)
            strategy: Function(df, current_step, portfolio_state) → action (0=hold, 1=buy, 2=sell)
            symbol: Ticker symbol for logging
            benchmark_col: Column to use for buy-and-hold benchmark

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        self._reset()
        logger.info(f"Starting backtest for {symbol}: {len(data)} bars, ${self.config.initial_balance:,.0f} initial")

        for step in range(len(data)):
            row = data.iloc[step]
            current_price = float(row["close"])
            current_time = data.index[step]

            # Build portfolio state for strategy
            portfolio_state = {
                "cash": self.cash,
                "position_qty": self.position_qty,
                "position_price": self.position_price,
                "position_side": self.position_side,
                "total_value": self._portfolio_value(current_price),
                "step": step,
            }

            # Get action from strategy
            action = strategy(data, step, portfolio_state)
            self.actions_log.append({"step": step, "action": action, "price": current_price})

            # Check stop loss / take profit before action
            if self.position_qty > 0:
                self._check_exit_rules(current_price, current_time, symbol)

            # Execute action
            if action == 1 and self.position_qty == 0:
                self._open_long(current_price, current_time, symbol)
            elif action == 2 and self.position_qty > 0:
                self._close_position(current_price, current_time, symbol)
            elif action == 2 and self.position_qty == 0 and self.config.allow_short:
                self._open_short(current_price, current_time, symbol)

            # Record equity
            self.equity_curve.append(self._portfolio_value(current_price))
            self.equity_dates.append(current_time)

        # Force close any open position
        if self.position_qty > 0:
            final_price = float(data.iloc[-1]["close"])
            self._close_position(final_price, data.index[-1], symbol)

        # Build results
        equity = pd.Series(self.equity_curve, index=self.equity_dates)
        trades_df = self._trades_to_dataframe()

        # Benchmark (buy & hold)
        benchmark = None
        if benchmark_col in data.columns:
            initial_shares = self.config.initial_balance / float(data[benchmark_col].iloc[0])
            benchmark = data[benchmark_col].astype(float) * initial_shares

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all(
            equity_curve=equity,
            trades=trades_df,
            benchmark=benchmark,
        )

        result = BacktestResult(
            symbol=symbol,
            config=self.config,
            equity_curve=equity,
            trades=trades_df,
            metrics=metrics,
            actions_log=pd.DataFrame(self.actions_log),
            benchmark=benchmark,
            start_date=data.index[0],
            end_date=data.index[-1],
        )

        logger.info(
            f"Backtest complete: return={metrics.total_return:.2%}, "
            f"sharpe={metrics.sharpe_ratio:.3f}, "
            f"trades={metrics.total_trades}, "
            f"win_rate={metrics.win_rate:.1%}"
        )

        return result

    # ── Trade Execution ───────────────────────────────────────

    def _open_long(self, price: float, time: Any, symbol: str):
        """Open a long position."""
        # Apply slippage
        fill_price = price * (1 + self.config.slippage)

        # Calculate position size
        available = self.cash * self.config.position_size
        commission = available * self.config.commission_rate
        qty = (available - commission) / fill_price

        self.position_qty = qty
        self.position_price = fill_price
        self.position_side = "LONG"
        self.cash -= (qty * fill_price + commission)

        self._entry_time = time
        self._entry_commission = commission

    def _open_short(self, price: float, time: Any, symbol: str):
        """Open a short position."""
        fill_price = price * (1 - self.config.slippage)
        available = self.cash * self.config.position_size
        commission = available * self.config.commission_rate
        qty = (available - commission) / fill_price

        self.position_qty = qty
        self.position_price = fill_price
        self.position_side = "SHORT"
        self.cash += (qty * fill_price - commission)

        self._entry_time = time
        self._entry_commission = commission

    def _close_position(self, price: float, time: Any, symbol: str):
        """Close current position and record the trade."""
        if self.position_qty == 0:
            return

        if self.position_side == "LONG":
            fill_price = price * (1 - self.config.slippage)
            proceeds = self.position_qty * fill_price
            commission = proceeds * self.config.commission_rate
            pnl = proceeds - commission - (self.position_qty * self.position_price + self._entry_commission)
            self.cash += proceeds - commission
        else:  # SHORT
            fill_price = price * (1 + self.config.slippage)
            cost = self.position_qty * fill_price
            commission = cost * self.config.commission_rate
            pnl = (self.position_qty * self.position_price) - cost - commission - self._entry_commission
            self.cash -= cost + commission

        pnl_pct = pnl / (self.position_qty * self.position_price)

        trade = TradeRecord(
            symbol=symbol,
            side=self.position_side,
            entry_price=self.position_price,
            exit_price=fill_price,
            quantity=self.position_qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission + self._entry_commission,
            entry_time=getattr(self, "_entry_time", None),
            exit_time=time,
        )
        self.trades.append(trade)

        # Reset position
        self.position_qty = 0.0
        self.position_price = 0.0
        self.position_side = None

    def _check_exit_rules(self, current_price: float, time: Any, symbol: str):
        """Check stop loss and take profit rules."""
        if self.position_qty == 0:
            return

        if self.position_side == "LONG":
            pnl_pct = (current_price - self.position_price) / self.position_price
        else:
            pnl_pct = (self.position_price - current_price) / self.position_price

        # Stop loss
        if self.config.stop_loss and pnl_pct <= -self.config.stop_loss:
            logger.debug(f"Stop loss triggered at {pnl_pct:.2%}")
            self._close_position(current_price, time, symbol)

        # Take profit
        elif self.config.take_profit and pnl_pct >= self.config.take_profit:
            logger.debug(f"Take profit triggered at {pnl_pct:.2%}")
            self._close_position(current_price, time, symbol)

    def _portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        if self.position_side == "LONG":
            return self.cash + self.position_qty * current_price
        elif self.position_side == "SHORT":
            return self.cash + self.position_qty * (2 * self.position_price - current_price)
        return self.cash

    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trade records to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "commission": t.commission,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in self.trades
        ])


@dataclass
class BacktestResult:
    """Complete backtest results container."""

    symbol: str
    config: BacktestConfig
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: PerformanceMetrics
    actions_log: pd.DataFrame
    benchmark: Optional[pd.Series] = None
    start_date: Any = None
    end_date: Any = None

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "symbol": self.symbol,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "initial_balance": self.config.initial_balance,
            "final_value": float(self.equity_curve.iloc[-1]),
            "metrics": self.metrics.to_dict(),
            "total_trades": len(self.trades),
            "equity_curve": {
                str(k): float(v) for k, v in self.equity_curve.items()
            },
            "trade_log": self.trades.to_dict(orient="records") if not self.trades.empty else [],
        }


# ── Benchmark Strategies ──────────────────────────────────────


def buy_and_hold_strategy(df: pd.DataFrame, step: int, portfolio: dict) -> int:
    """Buy and hold — buy on first bar, never sell."""
    if step == 0 and portfolio["position_qty"] == 0:
        return 1  # Buy
    return 0  # Hold


def sma_crossover_strategy(
    df: pd.DataFrame, step: int, portfolio: dict,
    fast_col: str = "sma_20", slow_col: str = "sma_50",
) -> int:
    """SMA crossover strategy — buy when fast > slow, sell when fast < slow."""
    if step < 1:
        return 0

    if fast_col not in df.columns or slow_col not in df.columns:
        return 0

    fast_now = df[fast_col].iloc[step]
    slow_now = df[slow_col].iloc[step]
    fast_prev = df[fast_col].iloc[step - 1]
    slow_prev = df[slow_col].iloc[step - 1]

    if pd.isna(fast_now) or pd.isna(slow_now):
        return 0

    # Golden cross → Buy
    if fast_now > slow_now and fast_prev <= slow_prev and portfolio["position_qty"] == 0:
        return 1
    # Death cross → Sell
    elif fast_now < slow_now and fast_prev >= slow_prev and portfolio["position_qty"] > 0:
        return 2

    return 0


# ── Quick Backtest Helper ─────────────────────────────────────


async def quick_backtest(
    symbol: str,
    strategy: Callable = None,
    period: str = "2y",
    initial_balance: float = 100_000,
    commission: float = 0.001,
) -> BacktestResult:
    """Run a quick backtest with default settings.

    Args:
        symbol: Ticker symbol
        strategy: Strategy function (default: SMA crossover)
        period: Historical data period
        initial_balance: Starting capital
        commission: Commission rate

    Returns:
        BacktestResult
    """
    # Fetch data
    fetcher = DataFetcherFactory.auto_detect(symbol)
    data = await fetcher.fetch_ohlcv(symbol, period=period)

    if data.empty:
        raise ValueError(f"No data available for {symbol}")

    # Add indicators
    calc = TechnicalIndicators()
    data = calc.add_all_indicators(data)

    # Setup engine
    config = BacktestConfig(
        initial_balance=initial_balance,
        commission_rate=commission,
    )
    engine = BacktestEngine(config)

    # Default strategy
    if strategy is None:
        strategy = sma_crossover_strategy

    result = engine.run(data, strategy, symbol)
    return result


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--period", type=str, default="2y", help="Data period")
    parser.add_argument("--balance", type=float, default=100000, help="Initial balance")
    parser.add_argument("--strategy", type=str, default="sma", choices=["sma", "hold"], help="Strategy")
    args = parser.parse_args()

    async def main():
        strat = sma_crossover_strategy if args.strategy == "sma" else buy_and_hold_strategy
        result = await quick_backtest(
            symbol=args.symbol,
            strategy=strat,
            period=args.period,
            initial_balance=args.balance,
        )
        print(result.metrics.summary())

    asyncio.run(main())
