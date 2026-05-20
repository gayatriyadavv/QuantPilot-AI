"""
QuantPilot AI — SQLAlchemy ORM Models

All database tables for the trading platform.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Market Data ───────────────────────────────────────────────


class Instrument(Base):
    """Tradeable asset (stock, crypto, etc.)."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="YAHOO")
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False, default="STOCK")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    ohlcv_data = relationship("OHLCVData", back_populates="instrument", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="instrument", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="instrument", cascade="all, delete-orphan")
    sentiment_scores = relationship("SentimentScore", back_populates="instrument", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", name="uq_symbol_exchange"),
    )

    def __repr__(self) -> str:
        return f"<Instrument {self.symbol} ({self.exchange})>"


class OHLCVData(Base):
    """OHLCV price data for an instrument."""

    __tablename__ = "ohlcv_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="1d")
    datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    instrument = relationship("Instrument", back_populates="ohlcv_data")
    indicators = relationship("TechnicalIndicator", back_populates="ohlcv_record", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("instrument_id", "timeframe", "datetime", name="uq_ohlcv_unique"),
        Index("ix_ohlcv_symbol_time", "instrument_id", "datetime"),
    )

    def __repr__(self) -> str:
        return f"<OHLCV {self.instrument_id} {self.datetime} C={self.close}>"


class TechnicalIndicator(Base):
    """Computed technical indicators for a given OHLCV record."""

    __tablename__ = "technical_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ohlcv_id: Mapped[int] = mapped_column(Integer, ForeignKey("ohlcv_data.id"), nullable=False)

    # Trend
    sma_20: Mapped[Optional[float]] = mapped_column(Float)
    sma_50: Mapped[Optional[float]] = mapped_column(Float)
    sma_200: Mapped[Optional[float]] = mapped_column(Float)
    ema_12: Mapped[Optional[float]] = mapped_column(Float)
    ema_26: Mapped[Optional[float]] = mapped_column(Float)

    # MACD
    macd: Mapped[Optional[float]] = mapped_column(Float)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float)
    macd_histogram: Mapped[Optional[float]] = mapped_column(Float)

    # Momentum
    rsi: Mapped[Optional[float]] = mapped_column(Float)
    stochastic_k: Mapped[Optional[float]] = mapped_column(Float)
    stochastic_d: Mapped[Optional[float]] = mapped_column(Float)
    cci: Mapped[Optional[float]] = mapped_column(Float)
    williams_r: Mapped[Optional[float]] = mapped_column(Float)
    roc: Mapped[Optional[float]] = mapped_column(Float)

    # Volatility
    bb_upper: Mapped[Optional[float]] = mapped_column(Float)
    bb_middle: Mapped[Optional[float]] = mapped_column(Float)
    bb_lower: Mapped[Optional[float]] = mapped_column(Float)
    atr: Mapped[Optional[float]] = mapped_column(Float)

    # Volume
    obv: Mapped[Optional[float]] = mapped_column(Float)
    volume_sma: Mapped[Optional[float]] = mapped_column(Float)

    # ADX
    adx: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    ohlcv_record = relationship("OHLCVData", back_populates="indicators")

    __table_args__ = (
        Index("ix_tech_ohlcv", "ohlcv_id"),
    )


# ── Trading ───────────────────────────────────────────────────


class Trade(Base):
    """Executed trade record."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY or SELL
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    pnl: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    commission: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    slippage: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    entry_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    strategy: Mapped[str] = mapped_column(String(50), default="rl_agent")
    agent_model: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    instrument = relationship("Instrument", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_time", "entry_time"),
        Index("ix_trades_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Trade {self.side} {self.quantity} @ {self.entry_price}>"


class Position(Base):
    """Current open position."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(Integer, ForeignKey("instruments.id"), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    avg_entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    stop_loss: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    take_profit: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    opened_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    side: Mapped[str] = mapped_column(String(5), default="LONG")  # LONG or SHORT

    # Relationships
    instrument = relationship("Instrument", back_populates="positions")

    def __repr__(self) -> str:
        return f"<Position {self.side} {self.quantity} @ {self.avg_entry_price}>"


class PortfolioSnapshot(Base):
    """Point-in-time portfolio state snapshot."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    total_value: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    cash_balance: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    invested_value: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    daily_return: Mapped[Optional[float]] = mapped_column(Float)
    cumulative_return: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    num_positions: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        Index("ix_portfolio_time", "snapshot_time"),
    )


# ── Sentiment ─────────────────────────────────────────────────


class SentimentScore(Base):
    """Sentiment analysis result for a financial headline."""

    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("instruments.id"))
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(Text)
    positive_score: Mapped[float] = mapped_column(Float, nullable=False)
    negative_score: Mapped[float] = mapped_column(Float, nullable=False)
    neutral_score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(10), nullable=False)  # positive, negative, neutral
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    instrument = relationship("Instrument", back_populates="sentiment_scores")

    __table_args__ = (
        Index("ix_sentiment_symbol", "instrument_id"),
        Index("ix_sentiment_time", "analyzed_at"),
    )


# ── RL Models & Backtesting ──────────────────────────────────


class RLModel(Base):
    """Trained RL model metadata."""

    __tablename__ = "rl_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(String(10), nullable=False)  # PPO, DQN, A2C
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)
    training_reward: Mapped[Optional[float]] = mapped_column(Float)
    validation_sharpe: Mapped[Optional[float]] = mapped_column(Float)
    validation_return: Mapped[Optional[float]] = mapped_column(Float)
    checkpoint_path: Mapped[str] = mapped_column(String(500), nullable=False)
    training_steps: Mapped[int] = mapped_column(Integer, default=0)
    training_duration_secs: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="TRAINING")  # TRAINING, READY, FAILED
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    backtest_results = relationship("BacktestResult", back_populates="model", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<RLModel {self.name} ({self.algorithm})>"


class BacktestResult(Base):
    """Backtest evaluation results."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("rl_models.id"))
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), default="1d")
    start_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Performance metrics
    total_return: Mapped[float] = mapped_column(Float, nullable=False)
    annualized_return: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Float)
    win_rate: Mapped[float] = mapped_column(Float, nullable=False)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_trade_duration: Mapped[Optional[float]] = mapped_column(Float)
    expectancy: Mapped[Optional[float]] = mapped_column(Float)

    # Detailed data (stored as JSON)
    equity_curve: Mapped[Optional[dict]] = mapped_column(JSON)
    trade_log: Mapped[Optional[dict]] = mapped_column(JSON)
    monthly_returns: Mapped[Optional[dict]] = mapped_column(JSON)

    # Benchmark comparison
    benchmark_return: Mapped[Optional[float]] = mapped_column(Float)
    alpha: Mapped[Optional[float]] = mapped_column(Float)
    beta: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    model = relationship("RLModel", back_populates="backtest_results")

    __table_args__ = (
        Index("ix_backtest_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<BacktestResult {self.name} return={self.total_return:.2%}>"
