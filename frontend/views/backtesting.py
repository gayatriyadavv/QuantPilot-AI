"""
QuantPilot AI — Backtesting Page

Run backtests, visualize equity curves, and compare strategies.
"""

from __future__ import annotations

import asyncio
from frontend.utils import run_async

import streamlit as st

from frontend.components.charts import (
    create_equity_curve,
    create_drawdown_chart,
    create_monthly_returns_heatmap,
)
from frontend.components.metrics import metric_row, render_trade_table


def render_backtesting():
    """Render the backtesting page."""
    st.markdown("## 🔬 Backtesting Engine")

    # ── Configuration ─────────────────────────────────────────
    with st.container():
        st.markdown("### ⚙️ Configuration")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            symbol = st.selectbox("Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "SPY"], key="bt_symbol")
        with col2:
            strategy = st.selectbox("Strategy", ["sma", "hold"], key="bt_strategy",
                                    format_func=lambda x: {"sma": "SMA Crossover", "hold": "Buy & Hold"}.get(x, x))
        with col3:
            period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], key="bt_period")
        with col4:
            initial_balance = st.number_input("Initial Balance ($)", value=100_000, step=10_000, key="bt_balance")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            commission = st.number_input("Commission (%)", value=0.1, step=0.01, key="bt_comm") / 100
        with col6:
            stop_loss = st.number_input("Stop Loss (%)", value=0.0, step=1.0, key="bt_sl")
            stop_loss = stop_loss / 100 if stop_loss > 0 else None
        with col7:
            take_profit = st.number_input("Take Profit (%)", value=0.0, step=1.0, key="bt_tp")
            take_profit = take_profit / 100 if take_profit > 0 else None
        with col8:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Backtest", use_container_width=True, type="primary", key="bt_run")

    # ── Run Backtest ──────────────────────────────────────────
    if run_btn:
        with st.spinner("Running backtest..."):
            try:
                from backend.backtesting.engine import BacktestConfig, BacktestEngine, sma_crossover_strategy, buy_and_hold_strategy
                from backend.data.fetcher import DataFetcherFactory
                from backend.indicators.technical import TechnicalIndicators

                # Fetch data
                fetcher = DataFetcherFactory.auto_detect(symbol)
                try:
                    df = run_async(fetcher.fetch_ohlcv(symbol, period=period))
                except Exception as e:
                    st.error(f"Failed to fetch data: {e}")
                    return

                if df.empty:
                    st.error(f"No data for {symbol}")
                    return

                calc = TechnicalIndicators()
                df = calc.add_all_indicators(df)

                # Select strategy
                strat_fn = sma_crossover_strategy if strategy == "sma" else buy_and_hold_strategy

                # Run
                config = BacktestConfig(
                    initial_balance=initial_balance,
                    commission_rate=commission,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                engine = BacktestEngine(config)
                result = engine.run(df, strat_fn, symbol)

                st.session_state["bt_result"] = result
                st.success("✅ Backtest complete!")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

    # ── Display Results ───────────────────────────────────────
    result = st.session_state.get("bt_result")

    if result is None:
        st.info("Configure parameters and click **Run Backtest** to see results.")
        return

    m = result.metrics
    st.markdown("---")

    # ── Performance KPIs ──────────────────────────────────────
    st.markdown("### 📊 Performance Summary")

    metric_row([
        {"label": "Total Return", "value": f"{m.total_return:.2%}", "delta": f"${result.equity_curve.iloc[-1] - result.config.initial_balance:,.0f}"},
        {"label": "Sharpe Ratio", "value": f"{m.sharpe_ratio:.3f}"},
        {"label": "Max Drawdown", "value": f"{m.max_drawdown:.2%}"},
        {"label": "Win Rate", "value": f"{m.win_rate:.1%}"},
    ])

    metric_row([
        {"label": "Total Trades", "value": str(m.total_trades)},
        {"label": "Profit Factor", "value": f"{m.profit_factor:.3f}"},
        {"label": "Annual Return", "value": f"{m.annualized_return:.2%}"},
        {"label": "Final Value", "value": f"${result.equity_curve.iloc[-1]:,.0f}"},
    ])

    # ── Equity Curve ──────────────────────────────────────────
    st.markdown("### 📈 Equity Curve")

    fig_eq = create_equity_curve(
        result.equity_curve,
        benchmark=result.benchmark,
        title=f"{result.symbol} — Equity Curve",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Drawdown ──────────────────────────────────────────────
    st.markdown("### 📉 Drawdown")
    fig_dd = create_drawdown_chart(result.equity_curve, height=250)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Detailed Metrics ──────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📋 Detailed Metrics")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Sortino Ratio | {m.sortino_ratio:.3f} |
        | Calmar Ratio | {m.calmar_ratio:.3f} |
        | Volatility (Ann.) | {m.volatility:.2%} |
        | VaR (95%) | {m.var_95:.2%} |
        | CVaR (95%) | {m.cvar_95:.2%} |
        | Max DD Duration | {m.max_drawdown_duration} days |
        | Avg Win | {m.avg_win:.4f} |
        | Avg Loss | {m.avg_loss:.4f} |
        | Expectancy | {m.expectancy:.4f} |
        | Benchmark Return | {m.benchmark_return:.2%} |
        | Alpha | {m.alpha:.4f} |
        | Beta | {m.beta:.4f} |
        """)

    with col_right:
        st.markdown("### 📊 Monthly Returns")
        if m.monthly_returns is not None and len(m.monthly_returns) > 0:
            from backend.backtesting.metrics import MetricsCalculator
            monthly_table = MetricsCalculator.monthly_return_table(result.equity_curve)
            if not monthly_table.empty:
                fig_hm = create_monthly_returns_heatmap(monthly_table, height=250)
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Insufficient data for monthly returns heatmap")
        else:
            st.info("Monthly return data not available")

    # ── Trade Log ─────────────────────────────────────────────
    st.markdown("### 📋 Trade Log")
    if not result.trades.empty:
        render_trade_table(result.trades.to_dict(orient="records"))
    else:
        st.info("No trades executed")

    # ── Export ────────────────────────────────────────────────
    with st.expander("💾 Export Results"):
        col_json, col_html = st.columns(2)
        with col_json:
            import json
            report_data = result.to_dict()
            st.download_button(
                "📥 Download JSON",
                json.dumps(report_data, indent=2, default=str),
                f"backtest_{result.symbol}.json",
                "application/json",
                use_container_width=True,
            )
        with col_html:
            st.info("HTML report: run `make backtest` from CLI")
