"""
QuantPilot AI — Portfolio Analytics Page

Portfolio overview, PnL tracking, position management, and risk metrics.
"""

from __future__ import annotations

import streamlit as st

from frontend.components.charts import create_allocation_chart, create_equity_curve
from frontend.components.metrics import metric_row, pnl_display, render_kpi_card


def render_portfolio():
    """Render the portfolio analytics page."""
    st.markdown("## 💼 Portfolio Analytics")

    # ── Portfolio Summary ─────────────────────────────────────
    st.markdown("### 📊 Portfolio Overview")

    # Simulated portfolio data
    total_value = 125_430.50
    cash = 34_250.00
    invested = total_value - cash
    unrealized_pnl = 5_430.50
    realized_pnl = 12_340.00
    daily_return = 0.0082

    metric_row([
        {"label": "Total Value", "value": f"${total_value:,.2f}", "delta": f"+{daily_return:.2%} today"},
        {"label": "Cash Available", "value": f"${cash:,.2f}"},
        {"label": "Invested", "value": f"${invested:,.2f}"},
        {"label": "Unrealized P&L", "value": pnl_display(unrealized_pnl)},
    ])

    metric_row([
        {"label": "Realized P&L", "value": pnl_display(realized_pnl)},
        {"label": "Total Return", "value": "+25.43%", "delta": "+$25,430"},
        {"label": "Sharpe Ratio", "value": "1.85"},
        {"label": "Max Drawdown", "value": "-8.3%"},
    ])

    st.markdown("---")

    # ── Two Column Layout ─────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # ── Positions ──
        st.markdown("### 📈 Open Positions")

        positions = [
            {"symbol": "AAPL", "side": "LONG", "qty": 150, "entry": 178.50, "current": 195.20, "pnl": 2505.00, "pnl_pct": 9.36},
            {"symbol": "GOOGL", "side": "LONG", "qty": 50, "entry": 142.30, "current": 148.75, "pnl": 322.50, "pnl_pct": 4.53},
            {"symbol": "MSFT", "side": "LONG", "qty": 80, "entry": 415.00, "current": 432.80, "pnl": 1424.00, "pnl_pct": 4.29},
            {"symbol": "NVDA", "side": "LONG", "qty": 30, "entry": 875.00, "current": 914.30, "pnl": 1179.00, "pnl_pct": 4.49},
        ]

        for pos in positions:
            pnl_color = "🟢" if pos["pnl"] >= 0 else "🔴"
            st.markdown(f"""
            <div style="
                background: linear-gradient(145deg, rgba(18,23,43,0.9), rgba(10,14,23,0.95));
                border: 1px solid rgba(42,47,69,0.8);
                border-radius: 10px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div>
                    <span style="color: #e4e6eb; font-weight: 700; font-size: 1rem;">{pos['symbol']}</span>
                    <span style="color: #8b8fa3; margin-left: 0.5rem;">{pos['side']} × {pos['qty']}</span>
                </div>
                <div style="text-align: right;">
                    <div style="color: #e4e6eb; font-family: 'JetBrains Mono', monospace;">${pos['current']:.2f}</div>
                    <div style="color: {'#00ff88' if pos['pnl'] >= 0 else '#ff4444'}; font-size: 0.85rem;">
                        {pnl_color} ${pos['pnl']:,.2f} ({pos['pnl_pct']:+.2f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # ── Allocation Chart ──
        st.markdown("### 🥧 Allocation")
        allocations = {
            "AAPL": 29350,
            "GOOGL": 7437,
            "MSFT": 34624,
            "NVDA": 27429,
            "Cash": cash,
        }
        fig_alloc = create_allocation_chart(allocations, height=300)
        st.plotly_chart(fig_alloc, use_container_width=True)

    # ── Risk Metrics ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Risk Metrics")

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        render_kpi_card("Value at Risk (95%)", "-2.3%", icon="📉")
    with r2:
        render_kpi_card("Beta", "1.12", icon="β")
    with r3:
        render_kpi_card("Volatility", "18.5%", icon="📊")
    with r4:
        render_kpi_card("Sortino", "2.15", icon="📈")
    with r5:
        render_kpi_card("Calmar", "3.06", icon="⚖️")

    # ── Trade History ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Recent Trades")

    trade_data = [
        {"Date": "2025-05-19", "Symbol": "TSLA", "Side": "SELL", "Qty": 25, "Price": "$245.30", "PnL": "$+1,230", "Status": "🟢 Closed"},
        {"Date": "2025-05-18", "Symbol": "AAPL", "Side": "BUY", "Qty": 50, "Price": "$178.50", "PnL": "-", "Status": "🔵 Open"},
        {"Date": "2025-05-17", "Symbol": "META", "Side": "SELL", "Qty": 30, "Price": "$512.40", "PnL": "$+890", "Status": "🟢 Closed"},
        {"Date": "2025-05-16", "Symbol": "GOOGL", "Side": "BUY", "Qty": 50, "Price": "$142.30", "PnL": "-", "Status": "🔵 Open"},
        {"Date": "2025-05-15", "Symbol": "AMZN", "Side": "SELL", "Qty": 20, "Price": "$189.70", "PnL": "$-340", "Status": "🔴 Loss"},
    ]

    import pandas as pd
    st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
