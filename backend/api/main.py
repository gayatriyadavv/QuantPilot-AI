"""
QuantPilot AI — FastAPI Application

Main application entry point with CORS, routers, and lifecycle management.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import get_settings
from backend.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown hooks."""
    # Startup
    logger.info("🚀 Starting QuantPilot AI API...")
    await init_db()
    logger.info("✅ Database initialized")
    yield
    # Shutdown
    await close_db()
    logger.info("👋 QuantPilot AI API shut down")


# ── Create Application ────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="QuantPilot AI",
    description="AI-powered stock trading platform using reinforcement learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register Routers ─────────────────────────────────────────

from backend.api.routes import agents, backtest, data, portfolio, sentiment, trading

app.include_router(data.router, prefix="/api/data", tags=["Market Data"])
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["Sentiment"])
app.include_router(agents.router, prefix="/api/agents", tags=["RL Agents"])


# ── Root & Health ─────────────────────────────────────────────


@app.get("/", tags=["System"])
async def root():
    """API root — system information."""
    return {
        "name": "QuantPilot AI",
        "version": "1.0.0",
        "description": "AI-powered trading platform",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    from backend.database import check_db_connection

    db_ok = await check_db_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now().isoformat(),
    }
