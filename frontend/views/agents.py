"""
QuantPilot AI — RL Agent Management Page

Train, evaluate, and manage RL trading agents.
"""

from __future__ import annotations

import asyncio
from frontend.utils import run_async
from datetime import datetime

import streamlit as st

from frontend.components.metrics import metric_row, status_indicator


def render_agents():
    """Render the AI agents management page."""
    st.markdown("## 🤖 AI Agent Management")

    # ── Training Configuration ────────────────────────────────
    st.markdown("### 🏋️ Train New Agent")

    with st.form("train_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY"], key="agent_sym")
        with col2:
            algorithm = st.selectbox("Algorithm", ["PPO", "DQN", "A2C"], key="agent_algo")
        with col3:
            timesteps = st.select_slider(
                "Training Steps",
                options=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
                value=100_000,
                key="agent_steps",
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            window_size = st.slider("Observation Window", 10, 60, 30, key="agent_window")
        with col5:
            balance = st.number_input("Initial Balance", value=100_000, step=10_000, key="agent_bal")
        with col6:
            data_period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1, key="agent_period")

        model_name = st.text_input("Model Name (optional)", placeholder="Auto-generated if empty", key="agent_name")

        submitted = st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True)

    if submitted:
        with st.spinner(f"Training {algorithm} on {symbol} for {timesteps:,} steps..."):
            try:
                from backend.rl.training import TrainingPipeline
                from backend.config import RLAlgorithm

                pipeline = TrainingPipeline(
                    symbol=symbol,
                    algorithm=RLAlgorithm(algorithm),
                    total_timesteps=timesteps,
                    window_size=window_size,
                    initial_balance=balance,
                    data_period=data_period,
                    model_name=model_name or None,
                )

                results = run_async(pipeline.run())

                st.session_state["last_training"] = results
                st.success(f"✅ Training complete! Model: {results['model_name']}")

            except Exception as e:
                st.error(f"Training failed: {e}")

    # ── Last Training Results ─────────────────────────────────
    last_training = st.session_state.get("last_training")
    if last_training:
        st.markdown("---")
        st.markdown("### 📊 Last Training Results")

        eval_results = last_training.get("evaluation", {})
        metric_row([
            {"label": "Algorithm", "value": last_training["algorithm"]},
            {"label": "Symbol", "value": last_training["symbol"]},
            {"label": "Training Time", "value": f"{last_training.get('training_time_secs', 0):.1f}s"},
            {"label": "Mean Return", "value": f"{eval_results.get('mean_return', 0):.2%}"},
        ])

        metric_row([
            {"label": "Mean Reward", "value": f"{eval_results.get('mean_reward', 0):.2f}"},
            {"label": "Mean Trades", "value": f"{eval_results.get('mean_trades', 0):.0f}"},
            {"label": "Train Data", "value": f"{last_training.get('train_data_size', 0)} bars"},
            {"label": "Model Path", "value": last_training.get("model_name", "N/A")},
        ])

    # ── Saved Models ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Saved Models")

    from backend.config import CHECKPOINTS_DIR

    if CHECKPOINTS_DIR.exists():
        models = [d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()]
        if models:
            for model_dir in sorted(models, reverse=True):
                model_file = model_dir / "model.zip"
                best_file = model_dir / "best_model.zip"
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6

                st.markdown(f"""
                <div style="
                    background: linear-gradient(145deg, rgba(18,23,43,0.9), rgba(10,14,23,0.95));
                    border: 1px solid rgba(42,47,69,0.8);
                    border-radius: 10px;
                    padding: 0.8rem 1rem;
                    margin-bottom: 0.5rem;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #00d4ff; font-weight: 700;">📦 {model_dir.name}</span>
                        </div>
                        <div style="color: #8b8fa3; font-size: 0.85rem;">
                            {'✅ Model' if model_file.exists() else '❌ No model'} |
                            {'⭐ Best' if best_file.exists() else ''} |
                            {size:.1f} MB
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trained models found. Train an agent to get started.")
    else:
        st.info("Checkpoints directory not found. Train an agent to create it.")

    # ── Algorithm Comparison ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Algorithm Comparison")

    st.markdown("""
    | Algorithm | Best For | Training Speed | Stability | Action Space |
    |-----------|----------|---------------|-----------|-------------|
    | **PPO** | Most tasks | ⚡ Medium | ✅ High | Discrete & Continuous |
    | **DQN** | Discrete actions | 🐢 Slow | ⚡ Medium | Discrete only |
    | **A2C** | Fast prototyping | 🚀 Fast | ⚠️ Lower | Discrete & Continuous |
    """)

    st.info("💡 **Tip**: Start with PPO — it's the most reliable for trading environments. Use DQN for strict Buy/Sell/Hold. A2C is fastest but less stable.")
