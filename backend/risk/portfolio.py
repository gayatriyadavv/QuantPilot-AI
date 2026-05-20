"""
QuantPilot AI — Portfolio Management

Tracks open positions, computes portfolio-level metrics (return, Sharpe,
drawdown, volatility), handles rebalancing, and creates snapshots for DB
persistence.

Usage::

    from backend.risk.portfolio import PortfolioManager

    pm = PortfolioManager(initial_cash=100_000.0)
    pm.track_position("AAPL", side="LONG", quantity=50, entry_price=175.0)
    pm.update_price("AAPL", 180.0)

    value = pm.get_portfolio_value()
    metrics = pm.get_portfolio_metrics()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from backend.config import get_settings


# ── Data Containers ──────────────────────────────────────────────────────


@dataclass
class PositionRecord:
    """In-memory representation of an open or historical position.

    Attributes:
        symbol: Ticker symbol.
        side: ``"LONG"`` or ``"SHORT"``.
        quantity: Number of units held.
        avg_entry_price: Weighted-average entry price.
        current_price: Latest observed market price.
        opened_at: Timestamp when the position was first opened.
        closed_at: Timestamp when the position was closed (``None`` if open).
        realized_pnl: Cumulative realised PnL from partial/full closes.
        commission_paid: Total commissions paid on this position.
    """

    symbol: str
    side: str
    quantity: float
    avg_entry_price: float
    current_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Original cost of the position."""
        return abs(self.quantity) * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealised profit or loss."""
        if self.side == "LONG":
            return (self.current_price - self.avg_entry_price) * abs(self.quantity)
        return (self.avg_entry_price - self.current_price) * abs(self.quantity)

    @property
    def total_pnl(self) -> float:
        """Realised + unrealised PnL."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def is_open(self) -> bool:
        """Whether the position is still open."""
        return self.closed_at is None and self.quantity != 0


@dataclass
class TradeRecord:
    """Immutable record of a single trade execution."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pnl: Optional[float] = None
    notes: Optional[str] = None


# ── Portfolio Manager ────────────────────────────────────────────────────


class PortfolioManager:
    """Manages a set of concurrent positions and computes portfolio analytics.

    Parameters:
        initial_cash: Starting cash balance.
        commission_rate: Per-trade commission as a fraction of order value.
            Defaults to the value in ``Settings``.
        slippage: Per-trade slippage as a fraction of price. Defaults to the
            value in ``Settings``.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission_rate: Optional[float] = None,
        slippage: Optional[float] = None,
    ) -> None:
        settings = get_settings()

        self._initial_cash: float = initial_cash
        self._cash: float = initial_cash
        self._commission_rate: float = (
            commission_rate if commission_rate is not None else settings.default_commission_rate
        )
        self._slippage: float = (
            slippage if slippage is not None else settings.default_slippage
        )

        # Open positions: symbol → PositionRecord
        self._positions: Dict[str, PositionRecord] = {}

        # History
        self._trade_history: List[TradeRecord] = []
        self._closed_positions: List[PositionRecord] = []

        # Equity snapshots for metric calculations
        self._equity_curve: List[Tuple[datetime, float]] = [
            (datetime.now(timezone.utc), initial_cash)
        ]

        # Peak value for drawdown tracking
        self._peak_value: float = initial_cash

        logger.info(
            "PortfolioManager initialised | cash={c:,.2f} | comm={cr:.4f} | slip={s:.4f}",
            c=initial_cash,
            cr=self._commission_rate,
            s=self._slippage,
        )

    # ────────────────────────────────────────────────────────────
    # Position Tracking
    # ────────────────────────────────────────────────────────────

    def track_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
    ) -> PositionRecord:
        """Open or add to a position.

        If a position in *symbol* already exists on the **same side**, the
        new shares are averaged in.  If a position exists on the **opposite
        side**, it is partially or fully closed first (netting).

        Args:
            symbol: Ticker symbol.
            side: ``"LONG"`` or ``"SHORT"``.
            quantity: Number of units.
            entry_price: Per-unit entry price (before slippage).

        Returns:
            The new or updated ``PositionRecord``.
        """
        # Apply slippage
        if side == "LONG":
            fill_price = entry_price * (1.0 + self._slippage)
        else:
            fill_price = entry_price * (1.0 - self._slippage)

        order_value = quantity * fill_price
        commission = order_value * self._commission_rate

        # Record the trade
        trade = TradeRecord(
            symbol=symbol,
            side="BUY" if side == "LONG" else "SELL",
            quantity=quantity,
            price=fill_price,
            commission=commission,
        )

        if symbol in self._positions:
            existing = self._positions[symbol]

            if existing.side == side:
                # Average into existing position
                total_qty = existing.quantity + quantity
                existing.avg_entry_price = (
                    (existing.avg_entry_price * existing.quantity + fill_price * quantity)
                    / total_qty
                )
                existing.quantity = total_qty
                existing.commission_paid += commission
                self._cash -= order_value + commission
                logger.info(
                    "Position averaged | {sym} {side} +{qty} @ {p:.4f} → "
                    "total {tq} @ {ap:.4f}",
                    sym=symbol,
                    side=side,
                    qty=quantity,
                    p=fill_price,
                    tq=total_qty,
                    ap=existing.avg_entry_price,
                )
                self._trade_history.append(trade)
                self._record_equity()
                return existing

            else:
                # Opposite side → netting / close
                if quantity >= existing.quantity:
                    # Full close of existing + possible new position
                    close_qty = existing.quantity
                    remaining = quantity - close_qty
                    pnl = self._compute_close_pnl(existing, fill_price, close_qty)
                    trade.pnl = pnl

                    existing.realized_pnl += pnl
                    existing.quantity = 0
                    existing.closed_at = datetime.now(timezone.utc)
                    self._cash += close_qty * fill_price - commission + pnl
                    self._closed_positions.append(copy.deepcopy(existing))
                    del self._positions[symbol]

                    logger.info(
                        "Position closed (netted) | {sym} | PnL={pnl:+,.2f}",
                        sym=symbol,
                        pnl=pnl,
                    )

                    if remaining > 0:
                        # Open a new position on the opposite side
                        self._trade_history.append(trade)
                        return self._open_new_position(
                            symbol, side, remaining, fill_price, commission=0.0
                        )

                    self._trade_history.append(trade)
                    self._record_equity()
                    return existing
                else:
                    # Partial close
                    pnl = self._compute_close_pnl(existing, fill_price, quantity)
                    trade.pnl = pnl
                    existing.realized_pnl += pnl
                    existing.quantity -= quantity
                    existing.commission_paid += commission
                    self._cash += quantity * fill_price - commission + pnl
                    logger.info(
                        "Position partially closed | {sym} -{qty} | PnL={pnl:+,.2f} | "
                        "remaining={rem}",
                        sym=symbol,
                        qty=quantity,
                        pnl=pnl,
                        rem=existing.quantity,
                    )
                    self._trade_history.append(trade)
                    self._record_equity()
                    return existing
        else:
            self._trade_history.append(trade)
            return self._open_new_position(symbol, side, quantity, fill_price, commission)

    def _open_new_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float,
    ) -> PositionRecord:
        """Helper: create a brand-new position."""
        order_value = quantity * fill_price
        self._cash -= order_value + commission

        pos = PositionRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            avg_entry_price=fill_price,
            current_price=fill_price,
            commission_paid=commission,
        )
        self._positions[symbol] = pos
        logger.info(
            "Position opened | {sym} {side} {qty} @ {p:.4f} | cost=${v:,.2f}",
            sym=symbol,
            side=side,
            qty=quantity,
            p=fill_price,
            v=order_value,
        )
        self._record_equity()
        return pos

    # ────────────────────────────────────────────────────────────
    # Close Position
    # ────────────────────────────────────────────────────────────

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        quantity: Optional[float] = None,
    ) -> Tuple[float, PositionRecord]:
        """Close an open position (fully or partially).

        Args:
            symbol: Ticker of the position to close.
            exit_price: Per-unit exit price (before slippage).
            quantity: Units to close; ``None`` = close entire position.

        Returns:
            ``(realised_pnl, position_record)``

        Raises:
            KeyError: If no open position exists for *symbol*.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol!r}.")

        pos = self._positions[symbol]

        # Apply slippage (exit)
        if pos.side == "LONG":
            fill_price = exit_price * (1.0 - self._slippage)
        else:
            fill_price = exit_price * (1.0 + self._slippage)

        close_qty = quantity if quantity is not None else pos.quantity
        close_qty = min(close_qty, pos.quantity)

        commission = close_qty * fill_price * self._commission_rate
        pnl = self._compute_close_pnl(pos, fill_price, close_qty)

        # Update cash
        self._cash += close_qty * fill_price - commission

        # Record trade
        trade = TradeRecord(
            symbol=symbol,
            side="SELL" if pos.side == "LONG" else "BUY",
            quantity=close_qty,
            price=fill_price,
            commission=commission,
            pnl=pnl,
        )
        self._trade_history.append(trade)

        pos.realized_pnl += pnl
        pos.commission_paid += commission
        pos.quantity -= close_qty

        if pos.quantity <= 0:
            pos.quantity = 0
            pos.closed_at = datetime.now(timezone.utc)
            self._closed_positions.append(copy.deepcopy(pos))
            del self._positions[symbol]
            logger.info(
                "Position fully closed | {sym} | PnL={pnl:+,.2f} | "
                "commission={c:.2f}",
                sym=symbol,
                pnl=pnl,
                c=commission,
            )
        else:
            logger.info(
                "Position partially closed | {sym} -{qty} @ {p:.4f} | "
                "PnL={pnl:+,.2f} | remaining={rem}",
                sym=symbol,
                qty=close_qty,
                p=fill_price,
                pnl=pnl,
                rem=pos.quantity,
            )

        self._record_equity()
        return pnl, pos

    @staticmethod
    def _compute_close_pnl(
        pos: PositionRecord, fill_price: float, qty: float
    ) -> float:
        """Compute realised PnL for closing *qty* units at *fill_price*."""
        if pos.side == "LONG":
            return (fill_price - pos.avg_entry_price) * qty
        return (pos.avg_entry_price - fill_price) * qty

    # ────────────────────────────────────────────────────────────
    # Price Updates
    # ────────────────────────────────────────────────────────────

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current market price for a position.

        Args:
            symbol: Ticker symbol.
            price: Latest market price.
        """
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Batch-update prices for multiple symbols.

        Args:
            prices: Mapping of symbol → latest price.
        """
        for sym, px in prices.items():
            self.update_price(sym, px)
        self._record_equity()

    # ────────────────────────────────────────────────────────────
    # Portfolio Value
    # ────────────────────────────────────────────────────────────

    def get_portfolio_value(self) -> float:
        """Compute current total portfolio value (cash + positions).

        Returns:
            Total equity value.
        """
        positions_value = sum(p.market_value for p in self._positions.values())
        return self._cash + positions_value

    @property
    def cash(self) -> float:
        """Available cash balance."""
        return self._cash

    @property
    def invested_value(self) -> float:
        """Total current market value of all open positions."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealised PnL across all open positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realised PnL from all closed positions."""
        return sum(p.realized_pnl for p in self._closed_positions)

    @property
    def positions(self) -> Dict[str, PositionRecord]:
        """Read-only view of open positions."""
        return dict(self._positions)

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    # ────────────────────────────────────────────────────────────
    # Portfolio Metrics
    # ────────────────────────────────────────────────────────────

    def get_portfolio_metrics(
        self, risk_free_rate: float = 0.04, periods_per_year: int = 252
    ) -> Dict[str, Any]:
        """Compute key portfolio performance and risk metrics.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio.
            periods_per_year: Trading periods per year (252 for daily).

        Returns:
            Dictionary containing:
                - ``total_value``: Current portfolio equity.
                - ``cash``: Cash balance.
                - ``invested``: Position market values.
                - ``total_return_pct``: Overall percentage return.
                - ``unrealized_pnl``: Unrealised PnL.
                - ``realized_pnl``: Realised PnL.
                - ``num_positions``: Open position count.
                - ``volatility``: Annualised portfolio volatility.
                - ``sharpe_ratio``: Annualised Sharpe ratio.
                - ``max_drawdown_pct``: Maximum drawdown observed.
                - ``win_rate``: Fraction of closed positions that were winners.
        """
        total_value = self.get_portfolio_value()
        total_return_pct = (total_value - self._initial_cash) / self._initial_cash

        # Daily returns from equity curve
        returns = self._compute_returns()

        # Annualised volatility
        vol = float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0

        # Sharpe ratio
        if vol > 0 and len(returns) > 1:
            mean_return = float(np.mean(returns))
            daily_rf = risk_free_rate / periods_per_year
            sharpe = (mean_return - daily_rf) / float(np.std(returns))
            sharpe *= np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        max_dd = self._compute_max_drawdown()

        # Win rate
        winners = [p for p in self._closed_positions if p.realized_pnl > 0]
        win_rate = len(winners) / len(self._closed_positions) if self._closed_positions else 0.0

        return {
            "total_value": round(total_value, 2),
            "cash": round(self._cash, 2),
            "invested": round(self.invested_value, 2),
            "total_return_pct": round(total_return_pct, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "num_positions": self.num_positions,
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": len(self._trade_history),
        }

    def _compute_returns(self) -> np.ndarray:
        """Compute periodic returns from the equity curve."""
        if len(self._equity_curve) < 2:
            return np.array([])
        values = np.array([v for _, v in self._equity_curve])
        returns = np.diff(values) / values[:-1]
        return returns

    def _compute_max_drawdown(self) -> float:
        """Compute maximum peak-to-trough drawdown from the equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0

        values = np.array([v for _, v in self._equity_curve])
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        return float(np.max(drawdowns))

    # ────────────────────────────────────────────────────────────
    # Position History
    # ────────────────────────────────────────────────────────────

    def get_position_history(self) -> List[Dict[str, Any]]:
        """Return a list of all historical trades.

        Returns:
            List of trade dictionaries, newest first.
        """
        history: List[Dict[str, Any]] = []
        for t in reversed(self._trade_history):
            history.append(
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": round(t.price, 4),
                    "commission": round(t.commission, 4),
                    "pnl": round(t.pnl, 2) if t.pnl is not None else None,
                    "timestamp": t.timestamp.isoformat(),
                    "notes": t.notes,
                }
            )
        return history

    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """Return a summary of all closed positions.

        Returns:
            List of position dictionaries, newest first.
        """
        result: List[Dict[str, Any]] = []
        for p in reversed(self._closed_positions):
            result.append(
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "avg_entry_price": round(p.avg_entry_price, 4),
                    "realized_pnl": round(p.realized_pnl, 2),
                    "commission_paid": round(p.commission_paid, 4),
                    "opened_at": p.opened_at.isoformat(),
                    "closed_at": p.closed_at.isoformat() if p.closed_at else None,
                }
            )
        return result

    # ────────────────────────────────────────────────────────────
    # Rebalancing
    # ────────────────────────────────────────────────────────────

    def rebalance(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Compute rebalancing orders to match target portfolio weights.

        This method does **not** execute trades; it returns a list of proposed
        orders that the caller should validate through ``RiskManager`` and
        then execute.

        Args:
            target_weights: Desired allocation fractions keyed by symbol.
                Must sum to ≤ 1.0 (remaining fraction stays in cash).
            current_prices: Latest market prices for all symbols involved.

        Returns:
            List of order dictionaries with keys: ``symbol``, ``side``,
            ``quantity``, ``price``.

        Raises:
            ValueError: If target weights sum to more than 1.0.
        """
        total_weight = sum(target_weights.values())
        if total_weight > 1.0 + 1e-9:
            raise ValueError(
                f"Target weights sum to {total_weight:.4f}; must be ≤ 1.0."
            )

        portfolio_value = self.get_portfolio_value()
        # Update prices first
        self.update_prices(current_prices)

        orders: List[Dict[str, Any]] = []

        # Calculate target value for each symbol
        all_symbols = set(list(target_weights.keys()) + list(self._positions.keys()))

        for symbol in all_symbols:
            target_wt = target_weights.get(symbol, 0.0)
            target_value = portfolio_value * target_wt
            price = current_prices.get(symbol)

            if price is None or price <= 0:
                logger.warning(
                    "Skipping {sym} in rebalance — no valid price available",
                    sym=symbol,
                )
                continue

            current_value = 0.0
            current_side = "LONG"
            if symbol in self._positions:
                pos = self._positions[symbol]
                current_value = pos.market_value
                current_side = pos.side

            diff_value = target_value - current_value

            if abs(diff_value) < price:
                # Less than 1 share difference — skip
                continue

            if diff_value > 0:
                qty = int(diff_value / price)
                if qty > 0:
                    orders.append(
                        {
                            "symbol": symbol,
                            "side": "BUY",
                            "quantity": qty,
                            "price": price,
                        }
                    )
            else:
                qty = int(abs(diff_value) / price)
                if qty > 0:
                    orders.append(
                        {
                            "symbol": symbol,
                            "side": "SELL",
                            "quantity": qty,
                            "price": price,
                        }
                    )

        logger.info(
            "Rebalance computed | {n} orders | portfolio=${pv:,.2f}",
            n=len(orders),
            pv=portfolio_value,
        )
        return orders

    # ────────────────────────────────────────────────────────────
    # Snapshot for DB Persistence
    # ────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Create a point-in-time portfolio snapshot suitable for DB storage.

        The returned dictionary maps directly to the fields of the
        ``PortfolioSnapshot`` ORM model.

        Returns:
            Dictionary with all snapshot fields.
        """
        metrics = self.get_portfolio_metrics()
        now = datetime.now(timezone.utc)

        snap = {
            "snapshot_time": now,
            "total_value": metrics["total_value"],
            "cash_balance": round(self._cash, 2),
            "invested_value": metrics["invested"],
            "unrealized_pnl": metrics["unrealized_pnl"],
            "realized_pnl": metrics["realized_pnl"],
            "daily_return": None,
            "cumulative_return": metrics["total_return_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown_pct"],
            "num_positions": metrics["num_positions"],
        }

        # Daily return (if we have at least 2 equity points)
        if len(self._equity_curve) >= 2:
            prev_val = self._equity_curve[-2][1]
            curr_val = self._equity_curve[-1][1]
            if prev_val > 0:
                snap["daily_return"] = round((curr_val - prev_val) / prev_val, 6)

        logger.info(
            "Portfolio snapshot | value={v:,.2f} | return={r:.2%} | "
            "sharpe={s:.2f} | dd={d:.2%}",
            v=snap["total_value"],
            r=snap["cumulative_return"] or 0,
            s=snap["sharpe_ratio"] or 0,
            d=snap["max_drawdown"] or 0,
        )
        return snap

    # ────────────────────────────────────────────────────────────
    # Internal Helpers
    # ────────────────────────────────────────────────────────────

    def _record_equity(self) -> None:
        """Append the current portfolio value to the equity curve."""
        value = self.get_portfolio_value()
        self._equity_curve.append((datetime.now(timezone.utc), value))
        self._peak_value = max(self._peak_value, value)

    def reset(self) -> None:
        """Reset the portfolio to its initial state.

        Useful for backtesting runs.
        """
        self._cash = self._initial_cash
        self._positions.clear()
        self._trade_history.clear()
        self._closed_positions.clear()
        self._equity_curve = [(datetime.now(timezone.utc), self._initial_cash)]
        self._peak_value = self._initial_cash
        logger.info("Portfolio reset to initial state | cash={c:,.2f}", c=self._initial_cash)

    def __repr__(self) -> str:
        return (
            f"<PortfolioManager value={self.get_portfolio_value():,.2f} "
            f"cash={self._cash:,.2f} positions={self.num_positions}>"
        )
