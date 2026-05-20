"""
QuantPilot AI — Database Engine & Session Management

Async SQLAlchemy setup with automatic PostgreSQL/SQLite detection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from loguru import logger
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import get_settings

# ── Engine Setup ──────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _create_engine() -> AsyncEngine:
    """Create async SQLAlchemy engine based on configured database URL."""
    settings = get_settings()
    url = settings.database_url

    # Engine kwargs differ between PostgreSQL and SQLite
    kwargs: dict = {
        "echo": settings.debug and not settings.is_production,
    }

    if settings.is_postgres:
        kwargs.update(
            {
                "pool_size": 20,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
            }
        )
    else:
        # SQLite — minimal pooling
        kwargs.update(
            {
                "connect_args": {"check_same_thread": False},
            }
        )

    engine = create_async_engine(url, **kwargs)
    logger.info(f"Database engine created: {'PostgreSQL' if settings.is_postgres else 'SQLite'}")
    return engine


def get_engine() -> AsyncEngine:
    """Get or create the async engine singleton."""
    global _engine
    if _engine is None:
        _engine = _create_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory singleton."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


# ── Session Context Manager ──────────────────────────────────


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session scope."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_session() as session:
        yield session


# ── Lifecycle ─────────────────────────────────────────────────


async def init_db() -> None:
    """Initialize database — create all tables."""
    from backend.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def close_db() -> None:
    """Shutdown database connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def check_db_connection() -> bool:
    """Health check — verify database connectivity."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
