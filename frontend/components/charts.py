"""
QuantPilot AI — Reusable Plotly Chart Components

TradingView-style candlestick charts, equity curves, drawdown charts, and more.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color Palette ─────────────────────────────────────────────

COLORS = {
    "bg": "#0a0e17",
    "paper": "#12172b",
    "grid": "#1e2340",
    "text": "#8b8fa3",
    "text_bright": "#e4e6eb",
    "up": "#00ff88",
    "down": "#ff4444",
    "cyan": "#00d4ff",
    "purple": "#7c3aed",
    "magenta": "#ff006e",
    "orange": "#ff8c00",
    "gold": "#ffd700",
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    paper_bgcolor=COLORS["paper"],
    plot_bgcolor=COLORS["bg"],
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text_bright"])),
    hoverlabel=dict(bgcolor=COLORS["paper"], font_color=COLORS["text_bright"]),
)


def _apply_layout(fig: go.Figure, title: str = "", height: int = 500) -> go.Figure:
    """Apply consistent dark theme to figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text_bright"])),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── Candlestick Chart ─────────────────────────────────────────


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    indicators: Optional[list[str]] = None,
    show_volume: bool = True,
    buy_signals: Optional[pd.Series] = None,
    sell_signals: Optional[pd.Series] = None,
    height: int = 600,
) -> go.Figure:
    """Create a TradingView-style candlestick chart with optional indicators and signals.

    Args:
        df: OHLCV DataFrame
        title: Chart title
        indicators: List of column names to overlay (e.g., ['sma_20', 'sma_50', 'bb_upper', 'bb_lower'])
        show_volume: Whether to show volume bars
        buy_signals: Series with True where buy signal occurs
        sell_signals: Series with True where sell signal occurs
        height: Chart height in pixels
    """
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            increasing_fillcolor=COLORS["up"],
            decreasing_fillcolor=COLORS["down"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Indicator overlays
    indicator_colors = [COLORS["cyan"], COLORS["purple"], COLORS["orange"], COLORS["magenta"], COLORS["gold"]]
    if indicators:
        for i, col in enumerate(indicators):
            if col in df.columns:
                color = indicator_colors[i % len(indicator_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col],
                        mode="lines",
                        name=col.upper(),
                        line=dict(color=color, width=1.5),
                        opacity=0.8,
                    ),
                    row=1, col=1,
                )

    # Buy/Sell signal markers
    if buy_signals is not None:
        buy_points = df[buy_signals]
        if len(buy_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index, y=buy_points["low"] * 0.995,
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(symbol="triangle-up", size=12, color=COLORS["up"], line=dict(width=1, color="white")),
                ),
                row=1, col=1,
            )

    if sell_signals is not None:
        sell_points = df[sell_signals]
        if len(sell_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index, y=sell_points["high"] * 1.005,
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(symbol="triangle-down", size=12, color=COLORS["down"], line=dict(width=1, color="white")),
                ),
                row=1, col=1,
            )

    # Volume bars
    if show_volume and "volume" in df.columns:
        colors = [COLORS["up"] if c >= o else COLORS["down"] for c, o in zip(df["close"], df["open"])]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["volume"],
                marker_color=colors,
                opacity=0.5,
                name="Volume",
                showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_xaxes(rangeslider_visible=False)
    _apply_layout(fig, title, height)

    return fig


# ── Equity Curve ──────────────────────────────────────────────


def create_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Value",
    height: int = 400,
) -> go.Figure:
    """Create equity curve chart with optional benchmark comparison."""
    fig = go.Figure()

    # Portfolio equity
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["cyan"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.05)",
        )
    )

    # Benchmark
    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index, y=benchmark.values,
                mode="lines",
                name="Benchmark (B&H)",
                line=dict(color=COLORS["purple"], width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
    )
    _apply_layout(fig, title, height)

    return fig


# ── Drawdown Chart ────────────────────────────────────────────


def create_drawdown_chart(
    equity: pd.Series,
    title: str = "Drawdown",
    height: int = 250,
) -> go.Figure:
    """Create drawdown chart."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["down"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 68, 68, 0.1)",
        )
    )

    fig.update_layout(yaxis_title="Drawdown (%)")
    _apply_layout(fig, title, height)

    return fig


# ── Indicator Subplots ────────────────────────────────────────


def create_indicator_chart(
    df: pd.DataFrame,
    indicator: str,
    title: str = "",
    height: int = 250,
) -> go.Figure:
    """Create a single indicator chart."""
    fig = go.Figure()

    if indicator == "rsi":
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI", line=dict(color=COLORS["cyan"], width=1.5)))
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["down"], opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["up"], opacity=0.5)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(0,212,255,0.03)", line_width=0)

    elif indicator == "macd":
        if "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color=COLORS["cyan"], width=1.5)))
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], mode="lines", name="Signal", line=dict(color=COLORS["orange"], width=1.5)))
            if "macd_histogram" in df.columns:
                colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in df["macd_histogram"]]
                fig.add_trace(go.Bar(x=df.index, y=df["macd_histogram"], marker_color=colors, opacity=0.5, name="Histogram"))

    elif indicator == "stochastic" and "stochastic_k" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["stochastic_k"], mode="lines", name="%K", line=dict(color=COLORS["cyan"], width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["stochastic_d"], mode="lines", name="%D", line=dict(color=COLORS["orange"], width=1.5)))
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS["down"], opacity=0.5)
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["up"], opacity=0.5)

    _apply_layout(fig, title or indicator.upper(), height)
    return fig


# ── Sentiment Gauge ───────────────────────────────────────────


def create_sentiment_gauge(
    score: float,
    label: str = "Sentiment",
    height: int = 250,
) -> go.Figure:
    """Create a sentiment gauge (bullish ↔ bearish)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title=dict(text=label, font=dict(size=14, color=COLORS["text_bright"])),
            number=dict(font=dict(color=COLORS["text_bright"], size=28)),
            gauge=dict(
                axis=dict(range=[-1, 1], tickcolor=COLORS["text"]),
                bar=dict(color=COLORS["cyan"]),
                bgcolor=COLORS["bg"],
                borderwidth=0,
                steps=[
                    dict(range=[-1, -0.3], color="rgba(255, 68, 68, 0.3)"),
                    dict(range=[-0.3, 0.3], color="rgba(139, 143, 163, 0.2)"),
                    dict(range=[0.3, 1], color="rgba(0, 255, 136, 0.3)"),
                ],
                threshold=dict(
                    line=dict(color=COLORS["text_bright"], width=2),
                    thickness=0.75,
                    value=score,
                ),
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# ── Portfolio Allocation Pie ──────────────────────────────────


def create_allocation_chart(
    allocations: dict[str, float],
    title: str = "Asset Allocation",
    height: int = 350,
) -> go.Figure:
    """Create portfolio allocation donut chart."""
    colors_cycle = [COLORS["cyan"], COLORS["purple"], COLORS["up"], COLORS["orange"], COLORS["magenta"], COLORS["gold"]]

    fig = go.Figure(
        go.Pie(
            labels=list(allocations.keys()),
            values=list(allocations.values()),
            hole=0.55,
            marker=dict(colors=colors_cycle[:len(allocations)], line=dict(color=COLORS["bg"], width=2)),
            textfont=dict(color=COLORS["text_bright"]),
        )
    )

    fig.update_layout(
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=title, font=dict(size=14, color=COLORS["text_bright"])),
        legend=dict(font=dict(color=COLORS["text_bright"])),
    )

    return fig


# ── Monthly Returns Heatmap ───────────────────────────────────


def create_monthly_returns_heatmap(
    monthly_returns: pd.DataFrame,
    title: str = "Monthly Returns (%)",
    height: int = 300,
) -> go.Figure:
    """Create a monthly returns heatmap."""
    fig = go.Figure(
        go.Heatmap(
            z=monthly_returns.values * 100,
            x=monthly_returns.columns.tolist(),
            y=monthly_returns.index.tolist(),
            colorscale=[
                [0, COLORS["down"]],
                [0.5, COLORS["bg"]],
                [1, COLORS["up"]],
            ],
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly_returns.values * 100],
            texttemplate="%{text}",
            textfont=dict(size=10, color=COLORS["text_bright"]),
        )
    )

    _apply_layout(fig, title, height)
    return fig
