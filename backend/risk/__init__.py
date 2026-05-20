"""
QuantPilot AI — Risk Management Module

Provides portfolio-level risk controls, position sizing, and drawdown management.

Components:
    - RiskManager: Core risk engine with stop-loss, take-profit, position sizing,
      and exposure limit checks.
    - PortfolioManager: Tracks positions, computes portfolio metrics, handles
      rebalancing and snapshots.
    - RiskConfig: Dataclass holding all configurable risk parameters.
"""

from backend.risk.manager import RiskConfig, RiskManager
from backend.risk.portfolio import PortfolioManager

__all__ = [
    "RiskConfig",
    "RiskManager",
    "PortfolioManager",
]
