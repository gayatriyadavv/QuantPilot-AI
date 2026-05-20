"""
QuantPilot AI — Metric Display Components

Reusable metric cards, KPIs, and formatted displays for Streamlit.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def metric_row(metrics: list[dict], columns: int = 4):
    """Display a row of metric cards.

    Args:
        metrics: List of dicts with keys: label, value, delta (optional), help (optional)
        columns: Number of columns
    """
    cols = st.columns(columns)
    for i, m in enumerate(metrics):
        with cols[i % columns]:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )


def pnl_display(value: float, prefix: str = "$", suffix: str = "") -> str:
    """Format PnL value with color markup."""
    if value >= 0:
        return f"🟢 {prefix}{value:,.2f}{suffix}"
    return f"🔴 {prefix}{value:,.2f}{suffix}"


def pct_display(value: float) -> str:
    """Format percentage with color indicator."""
    if value >= 0:
        return f"+{value:.2%}"
    return f"{value:.2%}"


def signal_badge(signal: str) -> str:
    """Create a colored signal badge."""
    color_map = {
        "BUY": "🟢",
        "STRONG_BUY": "🟢🟢",
        "SELL": "🔴",
        "STRONG_SELL": "🔴🔴",
        "HOLD": "🟡",
        "NEUTRAL": "⚪",
    }
    icon = color_map.get(signal.upper(), "⚪")
    return f"{icon} **{signal.upper()}**"


def status_indicator(status: str) -> str:
    """Create a status indicator."""
    indicators = {
        "READY": "🟢 Ready",
        "TRAINING": "🔵 Training",
        "FAILED": "🔴 Failed",
        "OPEN": "🟢 Open",
        "CLOSED": "⚪ Closed",
    }
    return indicators.get(status.upper(), f"⚪ {status}")


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    icon: str = "📊",
):
    """Render a styled KPI card using HTML."""
    delta_html = ""
    if delta:
        color = "#00ff88" if not delta.startswith("-") else "#ff4444"
        delta_html = f'<div style="color: {color}; font-size: 0.85rem; font-weight: 600;">{delta}</div>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, rgba(18,23,43,0.9), rgba(10,14,23,0.95));
        border: 1px solid rgba(42,47,69,0.8);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
    ">
        <div style="color: #8b8fa3; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">
            {icon} {label}
        </div>
        <div style="color: #e4e6eb; font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0.3rem 0;">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_trade_table(trades: list[dict]):
    """Render a styled trade history table."""
    if not trades:
        st.info("No trades to display")
        return

    import pandas as pd

    df = pd.DataFrame(trades)
    display_cols = ["symbol", "side", "entry_price", "exit_price", "pnl", "pnl_pct"]
    available = [c for c in display_cols if c in df.columns]

    if available:
        st.dataframe(
            df[available],
            use_container_width=True,
            hide_index=True,
            column_config={
                "pnl": st.column_config.NumberColumn("PnL", format="$%.2f"),
                "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
            },
        )
