# 🚀 QuantPilot AI

**AI-Powered Stock Trading Platform using Reinforcement Learning**

QuantPilot AI is a production-grade intelligent trading system that learns optimal buy/sell/hold strategies using historical market data, technical indicators, sentiment analysis, and reinforcement learning agents.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-orange?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Modules](#-modules)
- [API Documentation](#-api-documentation)
- [Training Pipeline](#-training-pipeline)
- [Backtesting](#-backtesting)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)

---

## ✨ Features

### Core
- 📊 **Historical Market Data** — Fetch from Yahoo Finance & Binance
- 📈 **25+ Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, ADX, and more
- 🤖 **RL Trading Agents** — PPO, DQN, A2C via Stable-Baselines3
- 🔬 **Backtesting Engine** — Full performance metrics with walk-forward analysis
- 💼 **Portfolio & Risk Management** — Stop-loss, position sizing, drawdown limits
- 📰 **AI Sentiment Analysis** — FinBERT-powered financial NLP
- 🖥️ **Premium Dashboard** — Dark-themed Streamlit UI with Plotly charts
- 📝 **Paper Trading** — Simulated trading environment

### Advanced
- 🧠 Composite reward functions (Sharpe, drawdown penalty, cost-aware)
- 📊 Multi-indicator consensus signal generation
- 🔄 Walk-forward training and evaluation
- 📈 Monthly returns heatmap & equity curve visualization
- 🐳 Docker Compose deployment with PostgreSQL & Redis
- 📋 REST API with full OpenAPI documentation

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│         (Charts, Portfolio, Backtesting, Agents)           │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼─────────────────────────────────────┐
│                     FastAPI Backend                         │
│    /api/data  /api/trading  /api/backtest  /api/agents     │
└──────┬───────────┬───────────┬───────────┬─────────────────┘
       │           │           │           │
┌──────▼──┐ ┌─────▼────┐ ┌───▼───┐ ┌─────▼──────┐
│  Data   │ │ Technical│ │  RL   │ │ Sentiment  │
│ Engine  │ │Indicators│ │Engine │ │  Analyzer  │
└────┬────┘ └──────────┘ └───┬───┘ └────────────┘
     │                       │
┌────▼───────────────────────▼───────────────────────────────┐
│              Database (PostgreSQL / SQLite)                 │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or uv package manager
- (Optional) Docker & Docker Compose

### 1. Clone & Install

```bash
cd "QuantPilot AI"

# Copy environment config
cp .env.example .env

# Install dependencies
pip install -e ".[dev]"
```

### 2. Start the API

```bash
make api
# or: uvicorn backend.api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 3. Start the Dashboard

```bash
make frontend
# or: streamlit run frontend/app.py --server.port 8501
```

Dashboard available at: http://localhost:8501

### 4. Train an RL Agent

```bash
# Quick training (10K steps)
make train-quick

# Full training
python -m backend.rl.training --symbol AAPL --algorithm PPO --timesteps 100000

# Fetch data first
make fetch-data
```

### 5. Run a Backtest

```bash
python -m backend.backtesting.engine --symbol AAPL --strategy sma
```

---

## 📁 Project Structure

```
QuantPilot AI/
├── backend/                  # Core Python backend
│   ├── api/                  # FastAPI application
│   │   ├── main.py           # App entry point, CORS, routers
│   │   ├── schemas.py        # Pydantic request/response models
│   │   └── routes/           # API route handlers
│   │       ├── data.py       # Market data endpoints
│   │       ├── trading.py    # Trading signal & execution
│   │       ├── backtest.py   # Backtesting endpoints
│   │       ├── portfolio.py  # Portfolio management
│   │       ├── sentiment.py  # Sentiment analysis
│   │       └── agents.py     # RL agent management
│   ├── backtesting/          # Backtesting engine
│   │   ├── engine.py         # Event-driven backtester
│   │   ├── metrics.py        # Performance metrics (Sharpe, etc.)
│   │   └── report.py         # JSON/HTML report generation
│   ├── data/                 # Data ingestion & storage
│   │   ├── fetcher.py        # Yahoo Finance & Binance fetchers
│   │   ├── processor.py      # Preprocessing & normalization
│   │   └── storage.py        # Database CRUD operations
│   ├── indicators/           # Technical analysis
│   │   ├── technical.py      # 25+ indicator calculations
│   │   └── signals.py        # Signal generation & consensus
│   ├── risk/                 # Risk management
│   │   ├── manager.py        # Stop-loss, position sizing, limits
│   │   └── portfolio.py      # Portfolio tracking & metrics
│   ├── rl/                   # Reinforcement learning
│   │   ├── environment.py    # Custom Gymnasium trading env
│   │   ├── rewards.py        # Modular reward functions
│   │   ├── agents.py         # PPO/DQN/A2C wrappers
│   │   ├── training.py       # End-to-end training pipeline
│   │   └── callbacks.py      # SB3 training callbacks
│   ├── sentiment/            # NLP sentiment analysis
│   │   ├── analyzer.py       # FinBERT sentiment classifier
│   │   └── news_fetcher.py   # Financial news collection
│   ├── config.py             # Central configuration (Pydantic)
│   ├── database.py           # Async SQLAlchemy engine
│   └── models.py             # ORM models (all tables)
├── frontend/                 # Streamlit dashboard
│   ├── app.py                # Main application entry
│   ├── styles.py             # Dark theme CSS
│   ├── components/           # Reusable UI components
│   │   ├── charts.py         # Plotly chart builders
│   │   └── metrics.py        # KPI cards & displays
│   └── pages/                # Dashboard pages
│       ├── dashboard.py      # Live charts & signals
│       ├── backtesting.py    # Backtest runner & results
│       ├── portfolio.py      # Portfolio analytics
│       ├── agents.py         # RL agent management
│       └── sentiment.py      # Sentiment analysis
├── docker/                   # Docker configuration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── models/                   # Saved RL model checkpoints
├── data/                     # Local data storage
├── tests/                    # Test suite
├── pyproject.toml            # Project dependencies
├── Makefile                  # Development commands
├── .env.example              # Environment template
└── README.md                 # This file
```

---

## 🧩 Modules

### Data Engine
- **Yahoo Finance**: Equities, ETFs (via `yfinance`)
- **Binance**: Crypto pairs (via `python-binance`)
- Auto-detection: Symbols ending in `USDT` route to Binance
- Missing value handling, normalization, feature engineering

### Technical Indicators
| Category | Indicators |
|----------|-----------|
| Trend | SMA(20,50,200), EMA(12,26), MACD, ADX |
| Momentum | RSI(14), Stochastic, CCI, Williams %R, ROC |
| Volatility | Bollinger Bands, ATR, Keltner Channels |
| Volume | OBV, Volume SMA, Volume Ratio |

### RL Trading Environment
- **Observation**: Window of normalized OHLCV + indicators + portfolio state
- **Actions**: Discrete(3) — Hold, Buy, Sell
- **Rewards**: Composite (LogReturn + Sharpe + Drawdown penalty + Cost penalty)
- **Algorithms**: PPO, DQN, A2C via Stable-Baselines3

### Risk Management
- Stop-loss (percentage & ATR-based)
- Take-profit (fixed & risk-reward ratio)
- Position sizing (Kelly criterion, volatility-adjusted)
- Max daily loss limits, drawdown limits
- Diversification constraints

### Sentiment Analysis
- **FinBERT** (`ProsusAI/finbert`) for financial NLP
- RSS feed parsing for financial news
- Yahoo Finance news integration
- Batch processing with caching

---

## 🔌 API Documentation

Start the API and visit http://localhost:8000/docs for interactive Swagger docs.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/data/ohlcv/{symbol}` | Fetch OHLCV data |
| `POST` | `/api/data/fetch` | Download & store data |
| `GET` | `/api/data/indicators/{symbol}` | Technical indicators |
| `POST` | `/api/trading/signal` | Get trading signal |
| `POST` | `/api/trading/execute` | Execute paper trade |
| `POST` | `/api/backtest/run` | Run backtest |
| `GET` | `/api/backtest/compare` | Compare strategies |
| `GET` | `/api/portfolio/summary` | Portfolio overview |
| `GET` | `/api/sentiment/{symbol}` | Symbol sentiment |
| `POST` | `/api/agents/train` | Start RL training |
| `GET` | `/api/agents/models` | List saved models |
| `POST` | `/api/agents/predict` | Get RL prediction |

---

## 🏋️ Training Pipeline

```bash
# Train with defaults (PPO, AAPL, 100K steps)
python -m backend.rl.training

# Custom training
python -m backend.rl.training \
    --symbol GOOGL \
    --algorithm DQN \
    --timesteps 250000 \
    --window 30 \
    --period 3y

# Or via API
curl -X POST http://localhost:8000/api/agents/train \
    -H "Content-Type: application/json" \
    -d '{"symbol": "AAPL", "algorithm": "PPO", "total_timesteps": 100000}'
```

The pipeline automatically:
1. Fetches historical data
2. Computes technical indicators
3. Creates the Gymnasium environment
4. Trains with composite reward function
5. Evaluates on held-out test data
6. Saves model checkpoint

---

## 🔬 Backtesting

```bash
# SMA Crossover strategy
python -m backend.backtesting.engine --symbol AAPL --strategy sma

# Buy & Hold benchmark
python -m backend.backtesting.engine --symbol AAPL --strategy hold
```

### Metrics Computed
- Total Return, Annualized Return, CAGR
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Max Drawdown Duration
- Win Rate, Profit Factor, Expectancy
- VaR (95%), CVaR (95%)
- Alpha, Beta (vs benchmark)

---

## 🐳 Docker Deployment

```bash
# Start all services
make docker-up
# or: docker compose -f docker/docker-compose.yml up --build -d

# View logs
make docker-logs

# Stop
make docker-down
```

Services:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Database (PostgreSQL for production, SQLite for dev)
DATABASE_URL=sqlite+aiosqlite:///./data/quantpilot.db

# API Keys (optional)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Trading defaults
DEFAULT_INITIAL_BALANCE=100000.0
DEFAULT_COMMISSION_RATE=0.001
MAX_DAILY_LOSS=0.05

# RL Training
DEFAULT_TRAINING_TIMESTEPS=100000
DEFAULT_ALGORITHM=PPO
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI, SQLAlchemy, Pydantic |
| **ML/RL** | PyTorch, Stable-Baselines3, Gymnasium |
| **Data** | yfinance, python-binance, pandas |
| **NLP** | HuggingFace Transformers, FinBERT |
| **Dashboard** | Streamlit, Plotly |
| **Database** | PostgreSQL (prod), SQLite (dev) |
| **Infrastructure** | Docker, Docker Compose |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>QuantPilot AI</b> — Where Reinforcement Learning Meets Financial Markets 🚀
</p>
