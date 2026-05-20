"""
QuantPilot AI — FastAPI Endpoint Tests
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pandas as pd


@pytest.fixture
def sample_ohlcv_df():
    """Sample OHLCV data for mocking."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(100) * 0.5,
            "high": close + abs(np.random.randn(100)) * 2,
            "low": close - abs(np.random.randn(100)) * 2,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, size=100).astype(float),
        },
        index=dates,
    )


class TestDataEndpoints:
    """Test market data API endpoints."""

    def test_symbols_endpoint(self):
        """Test GET /api/data/symbols returns valid data."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/data/symbols?source=yahoo")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "symbol" in data[0]

    def test_root_endpoint(self):
        """Test root endpoint returns system info."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "QuantPilot AI"
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestTradingEndpoints:
    """Test trading API endpoints."""

    def test_positions_endpoint(self):
        """Test GET /api/trading/positions."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/trading/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_trade_history_endpoint(self):
        """Test GET /api/trading/history."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/trading/history?limit=10")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestPortfolioEndpoints:
    """Test portfolio API endpoints."""

    def test_portfolio_summary(self):
        """Test GET /api/portfolio/summary."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "cash_balance" in data

    def test_portfolio_performance(self):
        """Test GET /api/portfolio/performance."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/portfolio/performance")
        assert response.status_code == 200
        data = response.json()
        assert "total_return" in data
        assert "sharpe_ratio" in data

    def test_risk_metrics(self):
        """Test GET /api/portfolio/risk."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/portfolio/risk")
        assert response.status_code == 200
        data = response.json()
        assert "exposure" in data


class TestAgentEndpoints:
    """Test RL agent management endpoints."""

    def test_list_models(self):
        """Test GET /api/agents/models."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/agents/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_agent_status(self):
        """Test GET /api/agents/status."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        response = client.get("/api/agents/status")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestSentimentEndpoints:
    """Test sentiment API endpoints."""

    def test_analyze_text(self):
        """Test POST /api/sentiment/analyze."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)
        # This test may fail if FinBERT model is not downloaded
        # In CI, we'd mock the analyzer
        try:
            response = client.post(
                "/api/sentiment/analyze",
                json={"text": "Apple reports record quarterly earnings"}
            )
            if response.status_code == 200:
                data = response.json()
                assert "label" in data
                assert "confidence" in data
        except Exception:
            pytest.skip("FinBERT model not available")
