"""
QuantPilot AI — Main Dashboard Page

Live charts, trading signals, indicator overlays, and portfolio summary.
"""

from __future__ import annotations

import asyncio
from frontend.utils import run_async

import pandas as pd
import streamlit as st

from frontend.components.charts import (
    create_candlestick_chart,
    create_indicator_chart,
)
from frontend.components.metrics import metric_row, signal_badge, render_kpi_card


def render_dashboard():
    """Render the main trading dashboard."""
    st.markdown("## 📈 Market Dashboard")

    # ── Symbol Selection ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        symbol = st.selectbox(
            "Symbol",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "SPY", "QQQ",
             "BTCUSDT", "ETHUSDT"],
            key="dash_symbol",
        )
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "5m", "1wk"], key="dash_tf")
    with col3:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "ytd"], key="dash_period")
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh", key="dash_refresh", use_container_width=True)

    # ── Fetch Data ────────────────────────────────────────────
    @st.cache_data(ttl=300, show_spinner="Fetching market data...")
    def load_data(sym: str, tf: str, per: str):
        from backend.data.fetcher import DataFetcherFactory
        from backend.indicators.technical import TechnicalIndicators
        from backend.indicators.signals import SignalGenerator

        fetcher = DataFetcherFactory.auto_detect(sym)
        try:
            df = run_async(fetcher.fetch_ohlcv(sym, timeframe=tf, period=per))
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return None, None, None

        if df.empty:
            return None, None, None

        calc = TechnicalIndicators()
        df_ind = calc.add_all_indicators(df)

        sig_gen = SignalGenerator()
        df_sig = sig_gen.consensus_signal(df_ind)
        latest_signals = sig_gen.get_latest_signals(df_sig)

        return df_ind, df_sig, latest_signals

    with st.spinner("Loading..."):
        df_ind, df_sig, latest_signals = load_data(symbol, timeframe, period)

    if df_ind is None or df_ind.empty:
        st.error(f"No data available for {symbol}")
        return

    # ── KPI Cards ─────────────────────────────────────────────
    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest
    price_change = float(latest["close"] - prev["close"])
    price_change_pct = price_change / float(prev["close"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Price", f"${latest['close']:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2%})")
    with c2:
        rsi_val = float(latest.get("rsi", 0))
        st.metric("RSI (14)", f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
    with c3:
        macd_val = float(latest.get("macd", 0))
        st.metric("MACD", f"{macd_val:.4f}")
    with c4:
        atr_val = float(latest.get("atr", 0))
        st.metric("ATR", f"{atr_val:.2f}")
    with c5:
        signal_text = latest_signals.get("consensus_signal", {})
        if isinstance(signal_text, dict):
            signal_label = signal_text.get("label", "NEUTRAL")
        else:
            signal_label = "NEUTRAL"
        st.metric("Signal", signal_label)

    # ── Price Chart ───────────────────────────────────────────
    st.markdown("### 📊 Price Chart")

    # Indicator overlay selection
    overlay_options = ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "bb_upper", "bb_lower", "bb_middle"]
    available_overlays = [o for o in overlay_options if o in df_ind.columns]
    selected_overlays = st.multiselect(
        "Indicator Overlays",
        available_overlays,
        default=["sma_20", "sma_50"] if "sma_20" in available_overlays else [],
        key="dash_overlays",
    )

    # Generate buy/sell signal markers
    buy_signals = None
    sell_signals = None
    if df_sig is not None and "consensus_signal" in df_sig.columns:
        buy_signals = df_sig["consensus_signal"].isin([1, 2])  # BUY or STRONG_BUY
        sell_signals = df_sig["consensus_signal"].isin([-1, -2])  # SELL or STRONG_SELL

    fig = create_candlestick_chart(
        df_ind,
        title=f"{symbol} — {timeframe.upper()}",
        indicators=selected_overlays,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Indicator Charts ──────────────────────────────────────
    st.markdown("### 📉 Technical Indicators")

    tab_rsi, tab_macd, tab_stoch = st.tabs(["RSI", "MACD", "Stochastic"])

    with tab_rsi:
        fig_rsi = create_indicator_chart(df_ind, "rsi", "RSI (14)", height=250)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_macd:
        fig_macd = create_indicator_chart(df_ind, "macd", "MACD", height=250)
        st.plotly_chart(fig_macd, use_container_width=True)

    with tab_stoch:
        fig_stoch = create_indicator_chart(df_ind, "stochastic", "Stochastic Oscillator", height=250)
        st.plotly_chart(fig_stoch, use_container_width=True)

    # ── Signal Summary ────────────────────────────────────────
    st.markdown("### 🎯 Signal Summary")

    if df_sig is not None:
        sig_cols = st.columns(5)
        signal_names = ["rsi_signal", "macd_signal_ind", "bollinger_signal", "sma_signal", "stochastic_signal"]
        signal_labels = ["RSI", "MACD", "Bollinger", "SMA Cross", "Stochastic"]

        for i, (col_name, label) in enumerate(zip(signal_names, signal_labels)):
            if col_name in df_sig.columns:
                val = int(df_sig[col_name].iloc[-1])
                signal_map = {-2: "STRONG_SELL", -1: "SELL", 0: "NEUTRAL", 1: "BUY", 2: "STRONG_BUY"}
                with sig_cols[i]:
                    st.markdown(f"**{label}**")
                    st.markdown(signal_badge(signal_map.get(val, "NEUTRAL")))

    # ── Data Table ────────────────────────────────────────────
    with st.expander("📋 Raw Data (Last 20 bars)"):
        display_cols = ["open", "high", "low", "close", "volume", "rsi", "macd", "atr"]
        available_cols = [c for c in display_cols if c in df_ind.columns]
        st.dataframe(
            df_ind[available_cols].tail(20).round(4),
            use_container_width=True,
        )
