"""
QuantPilot AI — Risk Management Engine

Centralised risk-control layer that validates every order before execution.
Provides stop-loss / take-profit management, position-sizing algorithms,
daily loss limits, drawdown checks, diversification rules, and exposure caps.

Usage::

    from backend.risk.manager import RiskManager, RiskConfig

    cfg = RiskConfig(max_position_pct=0.20, max_daily_loss_pct=0.03)
    rm  = RiskManager(config=cfg, initial_balance=100_000.0)

    # Validate a proposed order
    ok, reasons = rm.validate_order(
        symbol="AAPL",
        side="BUY",
        quantity=50,
        price=175.0,
        current_prices={"AAPL": 175.0, "MSFT": 320.0},
        portfolio_value=100_000.0,
        positions=current_positions,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from backend.config import get_settings


# ── Enums ────────────────────────────────────────────────────────────────


class PositionSizingMethod(str, Enum):
    """Supported position-sizing algorithms."""

    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


class StopLossType(str, Enum):
    """Supported stop-loss strategies."""

    PERCENTAGE = "percentage"
    ATR_TRAILING = "atr_trailing"


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class RiskConfig:
    """All tuneable risk-management parameters.

    Attributes:
        max_position_pct: Maximum fraction of portfolio allocated to a single
            position (0-1). Default pulled from ``Settings.max_position_size``.
        max_daily_loss_pct: Maximum daily loss as fraction of portfolio value.
        max_drawdown_pct: Maximum portfolio-level drawdown before trading halts.
        max_long_exposure_pct: Maximum total long exposure as fraction of
            portfolio value.
        max_short_exposure_pct: Maximum total short exposure as fraction of
            portfolio value.
        default_stop_loss_pct: Default percentage stop-loss distance (0-1).
        atr_stop_multiplier: ATR multiplier for ATR-based trailing stops.
        default_take_profit_pct: Default percentage take-profit distance (0-1).
        risk_reward_ratio: Minimum risk-reward ratio for take-profit targets.
        fixed_fraction: Fraction of equity risked per trade (fixed-fractional
            sizing).
        kelly_fraction: Kelly fraction cap to reduce over-betting.
        max_correlated_positions: Maximum number of highly-correlated positions.
        commission_rate: Assumed commission per trade value.
        slippage: Assumed slippage per trade value.
    """

    # ── Allocation limits ──
    max_position_pct: float = 0.0  # 0 → use Settings default
    max_daily_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.20

    # ── Exposure limits ──
    max_long_exposure_pct: float = 1.50
    max_short_exposure_pct: float = 0.50

    # ── Stop-loss ──
    default_stop_loss_pct: float = 0.05
    atr_stop_multiplier: float = 2.0

    # ── Take-profit ──
    default_take_profit_pct: float = 0.10
    risk_reward_ratio: float = 2.0

    # ── Position sizing ──
    fixed_fraction: float = 0.02
    kelly_fraction: float = 0.50  # Half-Kelly by default

    # ── Diversification ──
    max_correlated_positions: int = 5

    # ── Costs ──
    commission_rate: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        """Fill zero-valued fields from application settings."""
        settings = get_settings()
        if self.max_position_pct == 0.0:
            self.max_position_pct = settings.max_position_size
        if self.max_daily_loss_pct == 0.0:
            self.max_daily_loss_pct = settings.max_daily_loss
        if self.commission_rate == 0.0:
            self.commission_rate = settings.default_commission_rate
        if self.slippage == 0.0:
            self.slippage = settings.default_slippage


# ── In-memory position representation ───────────────────────────────────


@dataclass
class LivePosition:
    """Lightweight in-memory position used by the risk engine.

    This mirrors the DB ``Position`` model but lives in memory for fast
    access during order validation.
    """

    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    avg_entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    highest_price: float = 0.0  # for trailing stops
    lowest_price: float = float("inf")  # for short trailing stops

    # ── Derived properties ──

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
    def unrealized_pnl_pct(self) -> float:
        """Unrealised P&L as a fraction of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


# ── Risk Manager ─────────────────────────────────────────────────────────


class RiskManager:
    """Core risk-management engine.

    Responsible for:
    * Stop-loss and take-profit evaluation
    * Position-sizing (Kelly, fixed-fractional, volatility-adjusted)
    * Daily loss-limit tracking
    * Maximum-drawdown monitoring
    * Diversification and exposure-limit enforcement
    * Pre-trade order validation (``validate_order``)

    Parameters:
        config: A ``RiskConfig`` instance with all tuneable parameters.
        initial_balance: Starting portfolio equity (used for drawdown
            calculations).
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_balance: float = 100_000.0,
    ) -> None:
        self.config = config or RiskConfig()
        self.initial_balance = initial_balance

        # Drawdown tracking
        self._peak_portfolio_value: float = initial_balance

        # Daily PnL tracking  {date_str: cumulative_pnl}
        self._daily_pnl: Dict[str, float] = {}

        # Trading-halted flag
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None

        logger.info(
            "RiskManager initialised | balance={bal:,.2f} | max_pos={mp:.1%} | "
            "max_daily_loss={mdl:.1%} | max_dd={mdd:.1%}",
            bal=initial_balance,
            mp=self.config.max_position_pct,
            mdl=self.config.max_daily_loss_pct,
            mdd=self.config.max_drawdown_pct,
        )

    # ────────────────────────────────────────────────────────────
    # Stop-Loss Checks
    # ────────────────────────────────────────────────────────────

    def stop_loss_check(
        self,
        position: LivePosition,
        atr: Optional[float] = None,
        stop_type: StopLossType = StopLossType.PERCENTAGE,
    ) -> Tuple[bool, Optional[float]]:
        """Evaluate whether a position's stop-loss has been triggered.

        Supports two strategies:

        * **Percentage-based**: fixed distance from entry price.
        * **ATR trailing**: stop follows the high-water mark by
          ``atr_stop_multiplier × ATR``.

        Args:
            position: The position to evaluate.
            atr: Current Average True Range value (required for ATR trailing).
            stop_type: Which stop-loss strategy to use.

        Returns:
            A tuple ``(triggered, stop_price)`` where *triggered* is ``True``
            when the current price has breached the stop level.
        """
        if stop_type == StopLossType.PERCENTAGE:
            return self._percentage_stop(position)
        if stop_type == StopLossType.ATR_TRAILING:
            return self._atr_trailing_stop(position, atr)
        return False, None

    def _percentage_stop(self, position: LivePosition) -> Tuple[bool, Optional[float]]:
        """Fixed percentage-based stop-loss."""
        pct = self.config.default_stop_loss_pct

        if position.side == "LONG":
            stop_price = position.avg_entry_price * (1.0 - pct)
            triggered = position.current_price <= stop_price
        else:
            stop_price = position.avg_entry_price * (1.0 + pct)
            triggered = position.current_price >= stop_price

        if triggered:
            logger.warning(
                "STOP-LOSS triggered | {sym} {side} | price={p:.4f} | stop={s:.4f}",
                sym=position.symbol,
                side=position.side,
                p=position.current_price,
                s=stop_price,
            )
        return triggered, stop_price

    def _atr_trailing_stop(
        self,
        position: LivePosition,
        atr: Optional[float],
    ) -> Tuple[bool, Optional[float]]:
        """ATR-based trailing stop-loss.

        For LONG positions the stop trails below the highest observed price.
        For SHORT positions it trails above the lowest observed price.
        """
        if atr is None or atr <= 0:
            logger.warning("ATR not available for {sym}; falling back to percentage stop", sym=position.symbol)
            return self._percentage_stop(position)

        trail_distance = self.config.atr_stop_multiplier * atr

        if position.side == "LONG":
            # Update high-water mark
            position.highest_price = max(position.highest_price, position.current_price)
            stop_price = position.highest_price - trail_distance
            triggered = position.current_price <= stop_price
        else:
            position.lowest_price = min(position.lowest_price, position.current_price)
            stop_price = position.lowest_price + trail_distance
            triggered = position.current_price >= stop_price

        if triggered:
            logger.warning(
                "ATR TRAILING STOP triggered | {sym} {side} | price={p:.4f} | "
                "stop={s:.4f} | ATR={atr:.4f}",
                sym=position.symbol,
                side=position.side,
                p=position.current_price,
                s=stop_price,
                atr=atr,
            )
        return triggered, stop_price

    # ────────────────────────────────────────────────────────────
    # Take-Profit Checks
    # ────────────────────────────────────────────────────────────

    def take_profit_check(
        self,
        position: LivePosition,
        risk_reward: bool = False,
    ) -> Tuple[bool, Optional[float]]:
        """Evaluate whether a position has reached its take-profit target.

        Args:
            position: The position to evaluate.
            risk_reward: If ``True``, derive the target from the
                ``risk_reward_ratio`` and ``default_stop_loss_pct`` instead of
                using the fixed ``default_take_profit_pct``.

        Returns:
            ``(triggered, target_price)``
        """
        if risk_reward:
            return self._risk_reward_take_profit(position)
        return self._fixed_take_profit(position)

    def _fixed_take_profit(self, position: LivePosition) -> Tuple[bool, Optional[float]]:
        """Fixed percentage take-profit."""
        pct = self.config.default_take_profit_pct

        if position.side == "LONG":
            target = position.avg_entry_price * (1.0 + pct)
            triggered = position.current_price >= target
        else:
            target = position.avg_entry_price * (1.0 - pct)
            triggered = position.current_price <= target

        if triggered:
            logger.info(
                "TAKE-PROFIT reached | {sym} {side} | price={p:.4f} | target={t:.4f}",
                sym=position.symbol,
                side=position.side,
                p=position.current_price,
                t=target,
            )
        return triggered, target

    def _risk_reward_take_profit(self, position: LivePosition) -> Tuple[bool, Optional[float]]:
        """Take-profit derived from risk-reward ratio.

        Target distance = stop-loss distance × risk_reward_ratio.
        """
        risk_distance = position.avg_entry_price * self.config.default_stop_loss_pct
        reward_distance = risk_distance * self.config.risk_reward_ratio

        if position.side == "LONG":
            target = position.avg_entry_price + reward_distance
            triggered = position.current_price >= target
        else:
            target = position.avg_entry_price - reward_distance
            triggered = position.current_price <= target

        if triggered:
            logger.info(
                "R:R TAKE-PROFIT reached | {sym} {side} | price={p:.4f} | "
                "target={t:.4f} | R:R={rr:.1f}",
                sym=position.symbol,
                side=position.side,
                p=position.current_price,
                t=target,
                rr=self.config.risk_reward_ratio,
            )
        return triggered, target

    # ────────────────────────────────────────────────────────────
    # Position Sizing
    # ────────────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        portfolio_value: float,
        price: float,
        method: PositionSizingMethod = PositionSizingMethod.FIXED_FRACTIONAL,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        volatility: Optional[float] = None,
        target_risk_pct: Optional[float] = None,
    ) -> int:
        """Calculate the number of shares/units to trade.

        Three algorithms are supported:

        * **Fixed fractional** — risk a fixed fraction of equity.
        * **Kelly criterion** — optimal fraction based on win-rate and
          average win/loss ratio (capped by ``kelly_fraction``).
        * **Volatility adjusted** — scale size inversely with asset
          volatility so each position carries roughly equal risk.

        The result is always capped by ``max_position_pct``.

        Args:
            portfolio_value: Current total portfolio equity.
            price: Per-unit price of the asset.
            method: Sizing algorithm to use.
            win_rate: Historical win rate (Kelly only, 0-1).
            avg_win_loss_ratio: Avg-win / avg-loss (Kelly only).
            volatility: Annualised volatility (vol-adjusted only, 0-1).
            target_risk_pct: Target risk contribution per position (vol-adjusted).

        Returns:
            Integer number of shares/units (floored).

        Raises:
            ValueError: If required parameters for the chosen method are
                missing.
        """
        if price <= 0:
            logger.error("Invalid price {p} for position sizing", p=price)
            return 0

        if method == PositionSizingMethod.FIXED_FRACTIONAL:
            size = self._fixed_fractional_size(portfolio_value, price)
        elif method == PositionSizingMethod.KELLY_CRITERION:
            size = self._kelly_size(portfolio_value, price, win_rate, avg_win_loss_ratio)
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            size = self._volatility_adjusted_size(
                portfolio_value, price, volatility, target_risk_pct
            )
        else:
            logger.warning("Unknown sizing method {m}; defaulting to fixed-fractional", m=method)
            size = self._fixed_fractional_size(portfolio_value, price)

        # Hard cap: max_position_pct of portfolio
        max_units = int((portfolio_value * self.config.max_position_pct) / price)
        capped = min(size, max_units)
        capped = max(capped, 0)

        logger.debug(
            "Position size | method={m} | raw={raw} | capped={cap} | "
            "max_allowed={mx}",
            m=method.value,
            raw=size,
            cap=capped,
            mx=max_units,
        )
        return capped

    def _fixed_fractional_size(self, portfolio_value: float, price: float) -> int:
        """Risk ``fixed_fraction`` of equity per trade."""
        risk_amount = portfolio_value * self.config.fixed_fraction
        risk_per_share = price * self.config.default_stop_loss_pct
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)

    def _kelly_size(
        self,
        portfolio_value: float,
        price: float,
        win_rate: Optional[float],
        avg_win_loss_ratio: Optional[float],
    ) -> int:
        """Kelly criterion position sizing.

        ``f* = W - (1-W)/R`` where W = win rate, R = avg-win / avg-loss.
        Capped at ``kelly_fraction`` (half-Kelly by default).
        """
        if win_rate is None or avg_win_loss_ratio is None:
            raise ValueError(
                "Kelly criterion requires 'win_rate' and 'avg_win_loss_ratio'."
            )
        if avg_win_loss_ratio <= 0:
            return 0

        kelly_f = win_rate - (1.0 - win_rate) / avg_win_loss_ratio
        kelly_f = max(kelly_f, 0.0)
        kelly_f = min(kelly_f, self.config.kelly_fraction)  # cap

        allocation = portfolio_value * kelly_f
        return int(allocation / price)

    def _volatility_adjusted_size(
        self,
        portfolio_value: float,
        price: float,
        volatility: Optional[float],
        target_risk_pct: Optional[float],
    ) -> int:
        """Position size scaled inversely with volatility.

        Each position targets the same dollar-risk contribution:

        ``size = (portfolio_value × target_risk_pct) / (price × volatility)``
        """
        if volatility is None or volatility <= 0:
            raise ValueError(
                "Volatility-adjusted sizing requires a positive 'volatility'."
            )
        risk_pct = target_risk_pct if target_risk_pct else self.config.fixed_fraction
        dollar_risk = portfolio_value * risk_pct
        per_unit_risk = price * volatility
        return int(dollar_risk / per_unit_risk)

    # ────────────────────────────────────────────────────────────
    # Daily Loss Limit
    # ────────────────────────────────────────────────────────────

    def record_pnl(self, pnl: float, trade_date: Optional[date] = None) -> None:
        """Record a realised PnL amount for daily-loss tracking.

        Args:
            pnl: The realised profit (+) or loss (-) from a closed trade.
            trade_date: The date of the trade; defaults to today.
        """
        key = (trade_date or date.today()).isoformat()
        self._daily_pnl[key] = self._daily_pnl.get(key, 0.0) + pnl
        logger.debug("Recorded PnL {pnl:+.2f} on {d} | cumulative={c:+.2f}", pnl=pnl, d=key, c=self._daily_pnl[key])

    def check_daily_loss_limit(
        self,
        portfolio_value: float,
        check_date: Optional[date] = None,
    ) -> Tuple[bool, float]:
        """Check whether the daily loss limit has been breached.

        Args:
            portfolio_value: Current portfolio equity (for computing the
                threshold).
            check_date: Date to check; defaults to today.

        Returns:
            ``(breached, daily_pnl)`` — ``breached`` is ``True`` when daily
            losses have exceeded ``max_daily_loss_pct × portfolio_value``.
        """
        key = (check_date or date.today()).isoformat()
        daily_pnl = self._daily_pnl.get(key, 0.0)
        threshold = -abs(portfolio_value * self.config.max_daily_loss_pct)

        breached = daily_pnl <= threshold
        if breached:
            self._trading_halted = True
            self._halt_reason = (
                f"Daily loss limit breached: {daily_pnl:+,.2f} <= {threshold:+,.2f}"
            )
            logger.error(self._halt_reason)
        return breached, daily_pnl

    # ────────────────────────────────────────────────────────────
    # Drawdown Check
    # ────────────────────────────────────────────────────────────

    def check_max_drawdown(self, portfolio_value: float) -> Tuple[bool, float]:
        """Check whether portfolio-level maximum drawdown has been exceeded.

        Updates the internal high-water mark and computes the current drawdown.

        Args:
            portfolio_value: Current total portfolio value.

        Returns:
            ``(breached, current_drawdown_pct)``
        """
        self._peak_portfolio_value = max(self._peak_portfolio_value, portfolio_value)

        if self._peak_portfolio_value == 0:
            return False, 0.0

        drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value

        breached = drawdown >= self.config.max_drawdown_pct
        if breached:
            self._trading_halted = True
            self._halt_reason = (
                f"Max drawdown breached: {drawdown:.2%} >= {self.config.max_drawdown_pct:.2%}"
            )
            logger.error(self._halt_reason)
        else:
            logger.debug(
                "Drawdown check OK | dd={dd:.2%} | peak={pk:,.2f} | current={cv:,.2f}",
                dd=drawdown,
                pk=self._peak_portfolio_value,
                cv=portfolio_value,
            )
        return breached, drawdown

    # ────────────────────────────────────────────────────────────
    # Diversification Check
    # ────────────────────────────────────────────────────────────

    def check_diversification(
        self,
        symbol: str,
        order_value: float,
        portfolio_value: float,
        positions: Dict[str, LivePosition],
    ) -> Tuple[bool, str]:
        """Verify that a new/expanded position does not violate allocation limits.

        Checks:
        * Single-position concentration (``max_position_pct``).
        * Total number of correlated positions (``max_correlated_positions``).

        Args:
            symbol: Ticker of the asset being traded.
            order_value: Dollar value of the proposed order.
            portfolio_value: Current total portfolio equity.
            positions: All current open positions keyed by symbol.

        Returns:
            ``(passed, reason)`` — ``passed`` is ``True`` when the order
            complies with diversification rules.
        """
        if portfolio_value <= 0:
            return False, "Portfolio value must be positive."

        # Existing position value + proposed order
        existing_value = positions[symbol].market_value if symbol in positions else 0.0
        new_total = existing_value + order_value
        concentration = new_total / portfolio_value

        if concentration > self.config.max_position_pct:
            msg = (
                f"Concentration limit: {symbol} would be {concentration:.1%} of "
                f"portfolio (max {self.config.max_position_pct:.1%})."
            )
            logger.warning(msg)
            return False, msg

        # Too many open positions
        num_positions = len(positions) + (0 if symbol in positions else 1)
        if num_positions > self.config.max_correlated_positions:
            msg = (
                f"Too many positions: {num_positions} > "
                f"{self.config.max_correlated_positions} max."
            )
            logger.warning(msg)
            return False, msg

        return True, "Diversification OK."

    # ────────────────────────────────────────────────────────────
    # Exposure Limits
    # ────────────────────────────────────────────────────────────

    def check_exposure_limits(
        self,
        side: str,
        order_value: float,
        portfolio_value: float,
        positions: Dict[str, LivePosition],
    ) -> Tuple[bool, str]:
        """Ensure that total long/short exposure remains within bounds.

        Args:
            side: ``"BUY"`` (increases long exposure) or ``"SELL"`` (increases
                short exposure).
            order_value: Dollar value of the proposed order.
            portfolio_value: Current total portfolio equity.
            positions: All current open positions keyed by symbol.

        Returns:
            ``(passed, reason)``
        """
        long_exposure = sum(
            p.market_value for p in positions.values() if p.side == "LONG"
        )
        short_exposure = sum(
            p.market_value for p in positions.values() if p.side == "SHORT"
        )

        if side.upper() == "BUY":
            long_exposure += order_value
            max_pct = self.config.max_long_exposure_pct
            exposure_pct = long_exposure / portfolio_value if portfolio_value else 0
            label = "Long"
        else:
            short_exposure += order_value
            max_pct = self.config.max_short_exposure_pct
            exposure_pct = short_exposure / portfolio_value if portfolio_value else 0
            label = "Short"

        if exposure_pct > max_pct:
            msg = (
                f"{label} exposure limit: {exposure_pct:.1%} > {max_pct:.1%} max."
            )
            logger.warning(msg)
            return False, msg

        logger.debug(
            "Exposure OK | long={l:.1%} | short={s:.1%}",
            l=long_exposure / portfolio_value if portfolio_value else 0,
            s=short_exposure / portfolio_value if portfolio_value else 0,
        )
        return True, f"{label} exposure OK ({exposure_pct:.1%})."

    # ────────────────────────────────────────────────────────────
    # Unified Order Validation
    # ────────────────────────────────────────────────────────────

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        positions: Dict[str, LivePosition],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[str]]:
        """Run **all** risk checks before allowing a trade to execute.

        This is the primary entry point that the trading engine should call
        before placing any order.

        Checks performed:
        1. Trading-halted flag.
        2. Daily loss limit.
        3. Maximum drawdown.
        4. Diversification / concentration.
        5. Exposure limits.

        Args:
            symbol: Ticker symbol being traded.
            side: ``"BUY"`` or ``"SELL"``.
            quantity: Number of units.
            price: Expected execution price.
            portfolio_value: Current portfolio equity.
            positions: Dict of open ``LivePosition`` objects keyed by symbol.
            current_prices: Optional mapping of symbol → latest price (used to
                refresh position prices before checks).

        Returns:
            ``(approved, rejection_reasons)`` — ``approved`` is ``True`` only
            when **all** checks pass. ``rejection_reasons`` lists every rule
            that was violated.
        """
        reasons: List[str] = []

        # 0. Refresh position prices if provided
        if current_prices:
            for sym, pos in positions.items():
                if sym in current_prices:
                    pos.current_price = current_prices[sym]

        # 1. Trading halted?
        if self._trading_halted:
            reasons.append(f"Trading halted: {self._halt_reason}")
            logger.warning("Order rejected — trading halted | {r}", r=self._halt_reason)
            return False, reasons

        order_value = quantity * price

        # 2. Basic sanity
        if quantity <= 0 or price <= 0:
            reasons.append("Quantity and price must be positive.")

        # 3. Daily loss limit
        breached, daily_pnl = self.check_daily_loss_limit(portfolio_value)
        if breached:
            reasons.append(
                f"Daily loss limit breached (PnL today: {daily_pnl:+,.2f})."
            )

        # 4. Max drawdown
        dd_breached, dd_pct = self.check_max_drawdown(portfolio_value)
        if dd_breached:
            reasons.append(f"Max drawdown breached ({dd_pct:.2%}).")

        # 5. Diversification
        div_ok, div_msg = self.check_diversification(
            symbol, order_value, portfolio_value, positions
        )
        if not div_ok:
            reasons.append(div_msg)

        # 6. Exposure limits
        exp_ok, exp_msg = self.check_exposure_limits(
            side, order_value, portfolio_value, positions
        )
        if not exp_ok:
            reasons.append(exp_msg)

        approved = len(reasons) == 0
        if approved:
            logger.info(
                "Order APPROVED | {side} {qty} {sym} @ {p:.4f} (${val:,.2f})",
                side=side,
                qty=quantity,
                sym=symbol,
                p=price,
                val=order_value,
            )
        else:
            logger.warning(
                "Order REJECTED | {side} {qty} {sym} @ {p:.4f} | reasons={r}",
                side=side,
                qty=quantity,
                sym=symbol,
                p=price,
                r=reasons,
            )
        return approved, reasons

    # ────────────────────────────────────────────────────────────
    # Utility Helpers
    # ────────────────────────────────────────────────────────────

    def reset_daily_pnl(self, for_date: Optional[date] = None) -> None:
        """Reset accumulated daily PnL (e.g. at start of trading day)."""
        key = (for_date or date.today()).isoformat()
        self._daily_pnl[key] = 0.0
        logger.info("Daily PnL reset for {d}", d=key)

    def resume_trading(self) -> None:
        """Clear the trading-halted flag (manual override)."""
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Trading resumed (manual override)")

    @property
    def is_halted(self) -> bool:
        """Whether trading is currently halted."""
        return self._trading_halted

    @property
    def halt_reason(self) -> Optional[str]:
        """Reason trading was halted, or ``None``."""
        return self._halt_reason

    def get_daily_pnl(self, for_date: Optional[date] = None) -> float:
        """Return cumulative realised PnL for a given date."""
        key = (for_date or date.today()).isoformat()
        return self._daily_pnl.get(key, 0.0)

    def summary(self, portfolio_value: float) -> Dict[str, Any]:
        """Return a human-readable summary of the current risk state."""
        dd = 0.0
        if self._peak_portfolio_value > 0:
            dd = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value

        return {
            "is_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "peak_value": self._peak_portfolio_value,
            "current_drawdown_pct": round(dd, 4),
            "max_drawdown_limit_pct": self.config.max_drawdown_pct,
            "daily_pnl_today": self.get_daily_pnl(),
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "max_position_pct": self.config.max_position_pct,
            "max_long_exposure_pct": self.config.max_long_exposure_pct,
            "max_short_exposure_pct": self.config.max_short_exposure_pct,
        }
