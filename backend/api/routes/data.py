"""
QuantPilot AI — Market Data API Routes
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from backend.api.schemas import FetchDataRequest, IndicatorResponse, OHLCVResponse, SymbolInfo
from backend.data.fetcher import DataFetcherFactory, fetch_multiple_assets
from backend.database import get_session
from backend.data.storage import DataStorage
from backend.indicators.technical import TechnicalIndicators
from backend.indicators.signals import SignalGenerator

router = APIRouter()


@router.get("/symbols", response_model=list[SymbolInfo])
async def list_symbols(
    source: str = Query("yahoo", description="Data source: yahoo or binance"),
):
    """List available symbols from data source."""
    fetcher = DataFetcherFactory.get_fetcher(source)
    symbols = await fetcher.get_available_symbols()
    return [SymbolInfo(symbol=s, exchange=source.upper(), asset_type="STOCK" if source == "yahoo" else "CRYPTO") for s in symbols]


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query("1d", description="Candle interval"),
    period: str = Query("1y", description="Historical period"),
    source: str = Query("auto", description="Data source"),
):
    """Fetch OHLCV data for a symbol."""
    try:
        if source == "auto":
            fetcher = DataFetcherFactory.auto_detect(symbol)
        else:
            fetcher = DataFetcherFactory.get_fetcher(source)

        df = await fetcher.fetch_ohlcv(symbol, timeframe=timeframe, period=period)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        data = []
        for dt, row in df.iterrows():
            data.append({
                "datetime": str(dt),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": float(row["volume"]),
            })

        return OHLCVResponse(
            symbol=symbol,
            timeframe=timeframe,
            data_points=len(data),
            start_date=str(df.index[0]),
            end_date=str(df.index[-1]),
            data=data,
        )

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fetch")
async def fetch_and_store(request: FetchDataRequest):
    """Fetch data and store in database."""
    results = {}
    data = await fetch_multiple_assets(
        request.symbols, request.timeframe, request.period, request.source
    )

    async with get_session() as session:
        for symbol, df in data.items():
            if df.empty:
                results[symbol] = {"status": "no_data", "rows": 0}
                continue

            exchange = "BINANCE" if request.source == "binance" else "YAHOO"
            asset_type = "CRYPTO" if request.source == "binance" else "STOCK"

            instrument = await DataStorage.get_or_create_instrument(
                session, symbol, exchange=exchange, asset_type=asset_type,
            )
            rows = await DataStorage.save_ohlcv(session, instrument, df, request.timeframe)
            results[symbol] = {"status": "ok", "rows_added": rows, "total_rows": len(df)}

    return {"status": "ok", "results": results}


@router.get("/indicators/{symbol}", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str,
    timeframe: str = Query("1d"),
    period: str = Query("1y"),
):
    """Get technical indicators for a symbol."""
    try:
        fetcher = DataFetcherFactory.auto_detect(symbol)
        df = await fetcher.fetch_ohlcv(symbol, timeframe=timeframe, period=period)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        calc = TechnicalIndicators()
        df = calc.add_all_indicators(df)

        # Get latest values
        latest = df.iloc[-1]
        indicators = {}
        for col in df.columns:
            if col not in ["open", "high", "low", "close", "volume"]:
                val = latest[col]
                if not (isinstance(val, float) and (val != val)):  # not NaN
                    indicators[col] = round(float(val), 4)

        # Generate signals
        signal_gen = SignalGenerator()
        df_signals = signal_gen.consensus_signal(df)
        latest_signals = signal_gen.get_latest_signals(df_signals)

        return IndicatorResponse(
            symbol=symbol,
            indicators=indicators,
            latest_signals=latest_signals,
        )

    except Exception as e:
        logger.error(f"Error computing indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
