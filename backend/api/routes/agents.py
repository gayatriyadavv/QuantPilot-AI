"""
QuantPilot AI — RL Agent Management API Routes
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from backend.api.schemas import (
    AgentPredictionResponse,
    AgentStatusResponse,
    TrainAgentRequest,
)
from backend.config import CHECKPOINTS_DIR, RLAlgorithm

router = APIRouter()

# Track training jobs
_training_jobs: dict[str, dict] = {}


@router.post("/train")
async def train_agent(request: TrainAgentRequest, background_tasks: BackgroundTasks):
    """Start training an RL agent (runs in background)."""
    model_name = request.model_name or f"{request.algorithm}_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if model_name in _training_jobs and _training_jobs[model_name]["status"] == "TRAINING":
        raise HTTPException(status_code=409, detail=f"Model {model_name} is already training")

    _training_jobs[model_name] = {
        "model_name": model_name,
        "algorithm": request.algorithm,
        "symbol": request.symbol,
        "status": "TRAINING",
        "training_steps": request.total_timesteps,
        "started_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(_run_training, model_name, request)

    return {
        "status": "started",
        "model_name": model_name,
        "message": f"Training {request.algorithm} on {request.symbol} for {request.total_timesteps:,} steps",
    }


async def _run_training(model_name: str, request: TrainAgentRequest):
    """Background training task."""
    try:
        from backend.rl.training import TrainingPipeline

        pipeline = TrainingPipeline(
            symbol=request.symbol,
            algorithm=RLAlgorithm(request.algorithm),
            total_timesteps=request.total_timesteps,
            window_size=request.window_size,
            initial_balance=request.initial_balance,
            data_period=request.data_period,
            model_name=model_name,
        )

        results = await pipeline.run()

        _training_jobs[model_name].update({
            "status": "READY",
            "training_reward": results["evaluation"].get("mean_reward"),
            "validation_sharpe": None,
            "checkpoint_path": results["model_path"],
            "trained_at": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")
        _training_jobs[model_name].update({
            "status": "FAILED",
            "error": str(e),
        })


@router.get("/status", response_model=list[AgentStatusResponse])
async def get_agent_status():
    """Get status of all training jobs."""
    return [
        AgentStatusResponse(
            model_name=info["model_name"],
            algorithm=info["algorithm"],
            symbol=info["symbol"],
            status=info["status"],
            training_steps=info.get("training_steps", 0),
            training_reward=info.get("training_reward"),
            validation_sharpe=info.get("validation_sharpe"),
            checkpoint_path=info.get("checkpoint_path"),
            trained_at=info.get("trained_at"),
        )
        for info in _training_jobs.values()
    ]


@router.get("/models")
async def list_models():
    """List saved model checkpoints."""
    models = []
    checkpoints_dir = CHECKPOINTS_DIR
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                model_file = model_dir / "model.zip"
                best_file = model_dir / "best_model.zip"
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "has_model": model_file.exists(),
                    "has_best": best_file.exists(),
                    "size_mb": round(sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6, 2),
                })

    return {"models": models, "checkpoint_dir": str(checkpoints_dir)}


@router.post("/predict", response_model=AgentPredictionResponse)
async def get_prediction(model_name: str, symbol: str):
    """Get a prediction from a trained model."""
    try:
        model_path = CHECKPOINTS_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Load data and create environment
        from backend.data.fetcher import DataFetcherFactory
        from backend.indicators.technical import TechnicalIndicators
        from backend.rl.agents import TradingAgent
        from backend.rl.environment import TradingEnv

        fetcher = DataFetcherFactory.auto_detect(symbol)
        df = await fetcher.fetch_ohlcv(symbol, period="6mo")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        calc = TechnicalIndicators()
        df = calc.add_all_indicators(df)

        env = TradingEnv(df=df, window_size=30)
        agent = TradingAgent.load(str(model_path), env=env)

        obs, _ = env.reset()
        # Step to latest
        for _ in range(len(df) - env.window_size - 1):
            action, _ = agent.predict(obs)
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        action, _ = agent.predict(obs)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        return AgentPredictionResponse(
            model_name=model_name,
            symbol=symbol,
            action=action_map.get(action, "HOLD"),
            action_id=action,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
