.PHONY: install dev api frontend train test lint clean docker-up docker-down

# ── Installation ──────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	@echo "✅ QuantPilot AI installed successfully"

# ── Development ───────────────────────────────────────────────
dev: api

api:
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/app.py --server.port 8501 --theme.base dark

# Run both API and frontend
serve:
	@echo "Starting QuantPilot AI..."
	@make api &
	@sleep 2
	@make frontend

# ── Database ──────────────────────────────────────────────────
db-init:
	python -m backend.database init

db-migrate:
	alembic upgrade head

# ── Training ──────────────────────────────────────────────────
train:
	python -m backend.rl.training --symbol AAPL --algorithm PPO --timesteps 100000

train-quick:
	python -m backend.rl.training --symbol AAPL --algorithm PPO --timesteps 10000

backtest:
	python -m backend.backtesting.engine --symbol AAPL --model latest

# ── Data ──────────────────────────────────────────────────────
fetch-data:
	python -m backend.data.fetcher --symbols AAPL,GOOGL,MSFT,TSLA --period 2y

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=backend --cov-report=html

# ── Code Quality ──────────────────────────────────────────────
lint:
	ruff check backend/ frontend/ tests/

format:
	ruff format backend/ frontend/ tests/

# ── Docker ────────────────────────────────────────────────────
docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info
	@echo "🧹 Cleaned up"
