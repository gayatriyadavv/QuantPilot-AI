"""
QuantPilot AI — Central Configuration

All application settings managed via Pydantic BaseSettings with .env support.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project Paths ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for _dir in (DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Enums ─────────────────────────────────────────────────────


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class RLAlgorithm(str, Enum):
    PPO = "PPO"
    DQN = "DQN"
    A2C = "A2C"


class Timeframe(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"


# ── Settings ──────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──
    app_name: str = "QuantPilot AI"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"

    # ── Database ──
    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{DATA_DIR}/quantpilot.db",
        description="Async database URL (PostgreSQL or SQLite)",
    )

    # ── Server ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_url: str = "http://localhost:8501"
    backend_url: str = "http://localhost:8000"

    # ── API Keys ──
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    news_api_key: Optional[str] = None

    # ── Trading Defaults ──
    default_initial_balance: float = 100_000.0
    default_commission_rate: float = 0.001
    default_slippage: float = 0.0005
    max_position_size: float = 0.25
    max_daily_loss: float = 0.05

    # ── RL Training ──
    default_training_timesteps: int = 100_000
    default_window_size: int = 30
    default_algorithm: RLAlgorithm = RLAlgorithm.PPO
    model_checkpoint_dir: str = str(CHECKPOINTS_DIR)

    # ── Sentiment ──
    finbert_model: str = "ProsusAI/finbert"
    sentiment_cache_ttl: int = 3600

    # ── Redis ──
    redis_url: Optional[str] = None

    # ── Computed Properties ──

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url

    @property
    def is_postgres(self) -> bool:
        return "postgresql" in self.database_url

    @property
    def has_binance(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret)

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            return f"sqlite+aiosqlite:///{DATA_DIR}/quantpilot.db"
        return v


# ── Singleton ─────────────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
