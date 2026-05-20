"""
QuantPilot AI — Trading API Routes
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.api.schemas import (
    ExecuteTradeRequest,
    PositionResponse,
    TradeResponse,
    TradingSignalRequest,
    TradingSignalResponse,
)
from backend.data.fetcher import DataFetcherFactory
from backend.indicators.signals import SignalGenerator, SignalType
from backend.indicators.technical import TechnicalIndicators

router = APIRouter()

# In-memory paper trading state (would be DB-backed in production)
_paper_positions: list[dict] = []
_paper_trades: list[dict] = []
_trade_counter = 0


@router.post("/signal", response_model=TradingSignalResponse)
async def get_trading_signal(request: TradingSignalRequest):
    """Get current trading signal for a symbol."""
    try:
        fetcher = DataFetcherFactory.auto_detect(request.symbol)
        df = await fetcher.fetch_ohlcv(request.symbol, period="6mo")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        # Compute indicators and signals
        calc = TechnicalIndicators()
        df = calc.add_all_indicators(df)
        signal_gen = SignalGenerator()
        df = signal_gen.consensus_signal(df)

        latest = df.iloc[-1]
        consensus = int(latest.get("consensus_signal", 0))

        action_map = {
            SignalType.STRONG_BUY: "BUY",
            SignalType.BUY: "BUY",
            SignalType.NEUTRAL: "HOLD",
            SignalType.SELL: "SELL",
            SignalType.STRONG_SELL: "SELL",
        }

        action = action_map.get(SignalType(consensus), "HOLD")
        confidence = float(latest.get("signal_agreement", 0.5))

        indicators = {}
        for col in ["rsi", "macd", "adx", "atr", "bb_pct"]:
            if col in latest:
                indicators[col] = round(float(latest[col]), 4)

        return TradingSignalResponse(
            symbol=request.symbol,
            action=action,
            confidence=confidence,
            price=round(float(latest["close"]), 2),
            indicators=indicators,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions():
    """Get current open positions (paper trading)."""
    return [
        PositionResponse(**p) for p in _paper_positions if p.get("status") == "OPEN"
    ]


@router.post("/execute", response_model=TradeResponse)
async def execute_trade(request: ExecuteTradeRequest):
    """Execute a paper trade."""
    global _trade_counter

    try:
        fetcher = DataFetcherFactory.auto_detect(request.symbol)
        df = await fetcher.fetch_ohlcv(request.symbol, period="5d")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        price = request.price or float(df["close"].iloc[-1])
        quantity = request.quantity or 100.0

        _trade_counter += 1
        trade = {
            "id": _trade_counter,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": quantity,
            "entry_price": price,
            "exit_price": None,
            "pnl": None,
            "status": "OPEN",
            "entry_time": datetime.now().isoformat(),
            "exit_time": None,
        }

        _paper_trades.append(trade)
        _paper_positions.append({
            "symbol": request.symbol,
            "side": "LONG" if request.side == "BUY" else "SHORT",
            "quantity": quantity,
            "avg_entry_price": price,
            "current_price": price,
            "unrealized_pnl": 0.0,
            "status": "OPEN",
        })

        return TradeResponse(**trade)

    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=list[TradeResponse])
async def get_trade_history(limit: int = 50):
    """Get recent trade history."""
    return [TradeResponse(**t) for t in _paper_trades[-limit:]]
