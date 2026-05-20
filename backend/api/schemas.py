"""
QuantPilot AI — Pydantic Request/Response Schemas

All API data models for request validation and response serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Data Schemas ──────────────────────────────────────────────


class FetchDataRequest(BaseModel):
    symbols: list[str] = Field(..., description="List of ticker symbols", examples=[["AAPL", "GOOGL"]])
    timeframe: str = Field("1d", description="Candle interval")
    period: str = Field("2y", description="Historical period")
    source: str = Field("auto", description="Data source: yahoo, binance, auto")


class OHLCVResponse(BaseModel):
    symbol: str
    timeframe: str
    data_points: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data: list[dict[str, Any]]


class SymbolInfo(BaseModel):
    symbol: str
    exchange: str
    asset_type: str
    latest_close: Optional[float] = None
    data_points: Optional[int] = None


class IndicatorResponse(BaseModel):
    symbol: str
    indicators: dict[str, Any]
    latest_signals: dict[str, Any]


# ── Trading Schemas ───────────────────────────────────────────


class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    model_name: Optional[str] = Field(None, description="RL model to use")


class TradingSignalResponse(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    indicators: dict[str, Any] = {}
    timestamp: str


class ExecuteTradeRequest(BaseModel):
    symbol: str
    side: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: Optional[float] = None
    price: Optional[float] = None


class TradeResponse(BaseModel):
    id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str
    entry_time: str
    exit_time: Optional[str] = None


class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


# ── Backtest Schemas ──────────────────────────────────────────


class BacktestRequest(BaseModel):
    symbol: str = Field("AAPL", description="Ticker symbol")
    period: str = Field("2y", description="Data period")
    strategy: str = Field("sma", description="Strategy: sma, hold, rl")
    initial_balance: float = Field(100_000, description="Starting capital")
    commission_rate: float = Field(0.001, description="Commission rate")
    stop_loss: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit: Optional[float] = Field(None, description="Take profit percentage")
    model_name: Optional[str] = Field(None, description="RL model for rl strategy")


class BacktestResponse(BaseModel):
    id: Optional[int] = None
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_balance: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: dict[str, float]
    trade_log: list[dict[str, Any]]


# ── Portfolio Schemas ─────────────────────────────────────────


class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return: Optional[float] = None
    total_return: float
    positions: list[PositionResponse] = []


class PortfolioPerformance(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    total_trades: int
    equity_curve: dict[str, float]


# ── Sentiment Schemas ─────────────────────────────────────────


class SentimentAnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")


class SentimentResult(BaseModel):
    headline: str
    label: str
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float
    source: Optional[str] = None
    published_at: Optional[str] = None


class SentimentResponse(BaseModel):
    symbol: Optional[str] = None
    overall_sentiment: str
    sentiment_score: float
    results: list[SentimentResult]
    analyzed_at: str


# ── Agent Schemas ─────────────────────────────────────────────


class TrainAgentRequest(BaseModel):
    symbol: str = Field("AAPL", description="Ticker symbol")
    algorithm: str = Field("PPO", description="RL algorithm: PPO, DQN, A2C")
    total_timesteps: int = Field(100_000, description="Training timesteps")
    window_size: int = Field(30, description="Observation window")
    initial_balance: float = Field(100_000, description="Initial balance")
    data_period: str = Field("2y", description="Data period")
    model_name: Optional[str] = Field(None, description="Custom model name")


class AgentStatusResponse(BaseModel):
    model_name: str
    algorithm: str
    symbol: str
    status: str  # TRAINING, READY, FAILED
    training_steps: int = 0
    training_reward: Optional[float] = None
    validation_sharpe: Optional[float] = None
    checkpoint_path: Optional[str] = None
    trained_at: Optional[str] = None


class AgentPredictionResponse(BaseModel):
    model_name: str
    symbol: str
    action: str
    action_id: int
    timestamp: str


# ── Generic Schemas ───────────────────────────────────────────


class StatusResponse(BaseModel):
    status: str = "ok"
    message: str = ""
    timestamp: str = ""


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
