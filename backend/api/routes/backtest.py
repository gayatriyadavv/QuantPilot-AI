"""
QuantPilot AI — Backtesting API Routes
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.api.schemas import BacktestRequest, BacktestResponse
from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    buy_and_hold_strategy,
    sma_crossover_strategy,
)
from backend.data.fetcher import DataFetcherFactory
from backend.indicators.technical import TechnicalIndicators

router = APIRouter()

# Cache for backtest results
_backtest_cache: dict[int, dict] = {}
_bt_counter = 0


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run a backtest with specified parameters."""
    global _bt_counter

    try:
        # Fetch data
        fetcher = DataFetcherFactory.auto_detect(request.symbol)
        df = await fetcher.fetch_ohlcv(request.symbol, period=request.period)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        # Add indicators
        calc = TechnicalIndicators()
        df = calc.add_all_indicators(df)

        # Select strategy
        if request.strategy == "hold":
            strategy = buy_and_hold_strategy
        elif request.strategy == "sma":
            strategy = sma_crossover_strategy
        else:
            strategy = sma_crossover_strategy

        # Configure and run
        config = BacktestConfig(
            initial_balance=request.initial_balance,
            commission_rate=request.commission_rate,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
        )
        engine = BacktestEngine(config)
        result = engine.run(df, strategy, request.symbol)

        _bt_counter += 1
        m = result.metrics

        # Build equity curve (sampled for API response)
        equity = result.equity_curve
        step = max(1, len(equity) // 500)
        eq_dict = {str(k): round(float(v), 2) for k, v in equity.iloc[::step].items()}

        response = BacktestResponse(
            id=_bt_counter,
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=str(result.start_date),
            end_date=str(result.end_date),
            initial_balance=config.initial_balance,
            final_value=round(float(equity.iloc[-1]), 2),
            total_return=round(m.total_return, 6),
            annualized_return=round(m.annualized_return, 6),
            sharpe_ratio=round(m.sharpe_ratio, 4),
            sortino_ratio=round(m.sortino_ratio, 4),
            max_drawdown=round(m.max_drawdown, 6),
            win_rate=round(m.win_rate, 4),
            profit_factor=round(m.profit_factor, 4),
            total_trades=m.total_trades,
            equity_curve=eq_dict,
            trade_log=result.trades.to_dict(orient="records") if not result.trades.empty else [],
        )

        _backtest_cache[_bt_counter] = response.model_dump()
        return response

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: int):
    """Get results of a previous backtest."""
    if backtest_id not in _backtest_cache:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return _backtest_cache[backtest_id]


@router.get("/compare")
async def compare_strategies(
    symbol: str = "AAPL",
    period: str = "2y",
):
    """Compare SMA crossover vs Buy & Hold strategies."""
    try:
        fetcher = DataFetcherFactory.auto_detect(symbol)
        df = await fetcher.fetch_ohlcv(symbol, period=period)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        calc = TechnicalIndicators()
        df = calc.add_all_indicators(df)

        results = {}
        for name, strategy in [("sma_crossover", sma_crossover_strategy), ("buy_and_hold", buy_and_hold_strategy)]:
            engine = BacktestEngine()
            result = engine.run(df, strategy, symbol)
            m = result.metrics
            results[name] = {
                "total_return": round(m.total_return, 4),
                "sharpe_ratio": round(m.sharpe_ratio, 4),
                "max_drawdown": round(m.max_drawdown, 4),
                "win_rate": round(m.win_rate, 4),
                "total_trades": m.total_trades,
                "final_value": round(float(result.equity_curve.iloc[-1]), 2),
            }

        return {"symbol": symbol, "period": period, "strategies": results}

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
