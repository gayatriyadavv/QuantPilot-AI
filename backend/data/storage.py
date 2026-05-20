"""
QuantPilot AI — Data Storage Layer

CRUD operations for OHLCV data with database persistence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models import Instrument, OHLCVData


class DataStorage:
    """Database CRUD operations for market data."""

    # ── Instrument Operations ─────────────────────────────────

    @staticmethod
    async def get_or_create_instrument(
        session: AsyncSession,
        symbol: str,
        exchange: str = "YAHOO",
        asset_type: str = "STOCK",
        name: Optional[str] = None,
    ) -> Instrument:
        """Get an existing instrument or create a new one."""
        result = await session.execute(
            select(Instrument).where(
                and_(Instrument.symbol == symbol, Instrument.exchange == exchange)
            )
        )
        instrument = result.scalar_one_or_none()

        if instrument is None:
            instrument = Instrument(
                symbol=symbol,
                exchange=exchange,
                asset_type=asset_type,
                name=name or symbol,
            )
            session.add(instrument)
            await session.flush()
            logger.info(f"Created instrument: {symbol} ({exchange})")

        return instrument

    @staticmethod
    async def list_instruments(
        session: AsyncSession,
        asset_type: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> list[Instrument]:
        """List all registered instruments with optional filtering."""
        query = select(Instrument).where(Instrument.is_active == True)

        if asset_type:
            query = query.where(Instrument.asset_type == asset_type)
        if exchange:
            query = query.where(Instrument.exchange == exchange)

        result = await session.execute(query.order_by(Instrument.symbol))
        return list(result.scalars().all())

    # ── OHLCV Operations ──────────────────────────────────────

    @staticmethod
    async def save_ohlcv(
        session: AsyncSession,
        instrument: Instrument,
        df: pd.DataFrame,
        timeframe: str = "1d",
    ) -> int:
        """Bulk save OHLCV data with upsert logic (skip existing).

        Args:
            session: Database session
            instrument: Instrument record
            df: OHLCV DataFrame with DatetimeIndex
            timeframe: Candle timeframe

        Returns:
            Number of new rows inserted.
        """
        if df.empty:
            return 0

        # Get existing datetimes to avoid duplicates
        existing = await session.execute(
            select(OHLCVData.datetime).where(
                and_(
                    OHLCVData.instrument_id == instrument.id,
                    OHLCVData.timeframe == timeframe,
                )
            )
        )
        existing_dates = {row[0] for row in existing.fetchall()}

        # Insert only new records
        new_records = []
        for dt, row in df.iterrows():
            dt_val = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
            if dt_val in existing_dates:
                continue

            record = OHLCVData(
                instrument_id=instrument.id,
                timeframe=timeframe,
                datetime=dt_val,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            )
            new_records.append(record)

        if new_records:
            session.add_all(new_records)
            await session.flush()
            logger.info(
                f"Saved {len(new_records)} new OHLCV records for "
                f"{instrument.symbol} ({timeframe})"
            )

        return len(new_records)

    @staticmethod
    async def get_ohlcv(
        session: AsyncSession,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: str = "YAHOO",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query OHLCV data and return as DataFrame.

        Args:
            session: Database session
            symbol: Ticker symbol
            timeframe: Candle timeframe
            start_date: Filter from this date
            end_date: Filter to this date
            exchange: Exchange name
            limit: Max rows to return

        Returns:
            DataFrame with OHLCV columns and DatetimeIndex
        """
        # Find instrument
        instr_result = await session.execute(
            select(Instrument).where(
                and_(Instrument.symbol == symbol, Instrument.exchange == exchange)
            )
        )
        instrument = instr_result.scalar_one_or_none()

        if instrument is None:
            logger.warning(f"Instrument not found: {symbol} ({exchange})")
            return pd.DataFrame()

        # Build query
        query = (
            select(OHLCVData)
            .where(
                and_(
                    OHLCVData.instrument_id == instrument.id,
                    OHLCVData.timeframe == timeframe,
                )
            )
            .order_by(OHLCVData.datetime.asc())
        )

        if start_date:
            query = query.where(OHLCVData.datetime >= start_date)
        if end_date:
            query = query.where(OHLCVData.datetime <= end_date)
        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        records = result.scalars().all()

        if not records:
            return pd.DataFrame()

        # Convert to DataFrame
        data = [
            {
                "datetime": r.datetime,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in records
        ]

        df = pd.DataFrame(data)
        df.set_index("datetime", inplace=True)
        return df

    @staticmethod
    async def get_latest_date(
        session: AsyncSession,
        instrument_id: int,
        timeframe: str = "1d",
    ) -> Optional[datetime]:
        """Get the most recent data point date for an instrument."""
        result = await session.execute(
            select(OHLCVData.datetime)
            .where(
                and_(
                    OHLCVData.instrument_id == instrument_id,
                    OHLCVData.timeframe == timeframe,
                )
            )
            .order_by(OHLCVData.datetime.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row

    @staticmethod
    async def delete_ohlcv(
        session: AsyncSession,
        instrument_id: int,
        timeframe: Optional[str] = None,
    ) -> int:
        """Delete OHLCV data for an instrument."""
        conditions = [OHLCVData.instrument_id == instrument_id]
        if timeframe:
            conditions.append(OHLCVData.timeframe == timeframe)

        result = await session.execute(
            delete(OHLCVData).where(and_(*conditions))
        )
        count = result.rowcount
        logger.info(f"Deleted {count} OHLCV records for instrument {instrument_id}")
        return count

    # ── Export/Import ─────────────────────────────────────────

    @staticmethod
    async def export_to_csv(
        session: AsyncSession,
        symbol: str,
        filepath: str,
        timeframe: str = "1d",
        exchange: str = "YAHOO",
    ) -> bool:
        """Export OHLCV data to CSV file."""
        df = await DataStorage.get_ohlcv(session, symbol, timeframe, exchange=exchange)
        if df.empty:
            logger.warning(f"No data to export for {symbol}")
            return False

        df.to_csv(filepath)
        logger.info(f"Exported {len(df)} rows to {filepath}")
        return True

    @staticmethod
    def import_from_csv(filepath: str) -> pd.DataFrame:
        """Import OHLCV data from CSV file."""
        df = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")

        # Ensure expected columns
        expected = {"open", "high", "low", "close", "volume"}
        available = set(df.columns) & expected
        if available != expected:
            missing = expected - available
            logger.warning(f"CSV missing columns: {missing}")

        return df

    # ── Statistics ────────────────────────────────────────────

    @staticmethod
    async def get_data_stats(
        session: AsyncSession,
        symbol: str,
        timeframe: str = "1d",
        exchange: str = "YAHOO",
    ) -> dict:
        """Get summary statistics for stored data."""
        df = await DataStorage.get_ohlcv(session, symbol, timeframe, exchange=exchange)

        if df.empty:
            return {"symbol": symbol, "rows": 0}

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(df),
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "latest_close": float(df["close"].iloc[-1]),
            "avg_volume": float(df["volume"].mean()),
            "price_range": {
                "min": float(df["close"].min()),
                "max": float(df["close"].max()),
            },
        }
