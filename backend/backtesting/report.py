"""
QuantPilot AI — Backtest Report Generator

Generate structured reports from backtest results in JSON and HTML formats.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger


class ReportGenerator:
    """Generate formatted reports from backtest results."""

    @staticmethod
    def generate_json_report(
        result: Any,
        filepath: Optional[str] = None,
    ) -> dict:
        """Generate a JSON-serializable report.

        Args:
            result: BacktestResult object
            filepath: Optional path to save JSON file

        Returns:
            Report dictionary
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "symbol": result.symbol,
                "start_date": str(result.start_date),
                "end_date": str(result.end_date),
                "initial_balance": result.config.initial_balance,
                "commission_rate": result.config.commission_rate,
                "slippage": result.config.slippage,
            },
            "performance": {
                "final_value": float(result.equity_curve.iloc[-1]),
                "total_return": result.metrics.total_return,
                "annualized_return": result.metrics.annualized_return,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "calmar_ratio": result.metrics.calmar_ratio,
                "max_drawdown": result.metrics.max_drawdown,
                "volatility": result.metrics.volatility,
                "var_95": result.metrics.var_95,
                "cvar_95": result.metrics.cvar_95,
            },
            "trades": {
                "total_trades": result.metrics.total_trades,
                "winning_trades": result.metrics.winning_trades,
                "losing_trades": result.metrics.losing_trades,
                "win_rate": result.metrics.win_rate,
                "profit_factor": result.metrics.profit_factor,
                "expectancy": result.metrics.expectancy,
                "avg_win": result.metrics.avg_win,
                "avg_loss": result.metrics.avg_loss,
                "largest_win": result.metrics.largest_win,
                "largest_loss": result.metrics.largest_loss,
                "avg_trade_duration_days": result.metrics.avg_trade_duration,
            },
            "benchmark": {
                "benchmark_return": result.metrics.benchmark_return,
                "alpha": result.metrics.alpha,
                "beta": result.metrics.beta,
            },
            "trade_log": (
                result.trades.to_dict(orient="records") if not result.trades.empty else []
            ),
        }

        # Add monthly returns if available
        if result.metrics.monthly_returns is not None:
            report["monthly_returns"] = {
                str(k): float(v) for k, v in result.metrics.monthly_returns.items()
            }

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"JSON report saved to {filepath}")

        return report

    @staticmethod
    def generate_html_report(
        result: Any,
        filepath: str,
    ) -> str:
        """Generate an HTML report with styled tables and charts data.

        Args:
            result: BacktestResult object
            filepath: Path to save HTML file

        Returns:
            HTML string
        """
        m = result.metrics

        # Color helpers
        def pnl_color(val: float) -> str:
            return "#00ff88" if val >= 0 else "#ff4444"

        def pct_fmt(val: float) -> str:
            return f"{val:+.2%}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantPilot AI — Backtest Report: {result.symbol}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0a0e17;
            color: #e4e6eb;
            padding: 2rem;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, #1a1f35 0%, #0d1321 100%);
            border-radius: 16px;
            border: 1px solid #2a2f45;
        }}
        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{ color: #8b8fa3; margin-top: 0.5rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .card {{
            background: #12172b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #1e2340;
        }}
        .card .label {{ font-size: 0.85rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #12172b;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        th {{
            background: #1a1f35;
            padding: 0.8rem 1rem;
            text-align: left;
            font-size: 0.85rem;
            color: #8b8fa3;
            text-transform: uppercase;
        }}
        td {{
            padding: 0.8rem 1rem;
            border-top: 1px solid #1e2340;
        }}
        .section-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #00d4ff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantPilot AI</h1>
        <div class="subtitle">
            Backtest Report — {result.symbol} |
            {result.start_date} → {result.end_date}
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Total Return</div>
            <div class="value {'positive' if m.total_return >= 0 else 'negative'}">
                {pct_fmt(m.total_return)}
            </div>
        </div>
        <div class="card">
            <div class="label">Sharpe Ratio</div>
            <div class="value {'positive' if m.sharpe_ratio >= 1 else 'negative' if m.sharpe_ratio < 0 else ''}">
                {m.sharpe_ratio:.3f}
            </div>
        </div>
        <div class="card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{pct_fmt(m.max_drawdown)}</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value">{m.win_rate:.1%}</div>
        </div>
        <div class="card">
            <div class="label">Total Trades</div>
            <div class="value">{m.total_trades}</div>
        </div>
        <div class="card">
            <div class="label">Profit Factor</div>
            <div class="value {'positive' if m.profit_factor >= 1 else 'negative'}">
                {m.profit_factor:.3f}
            </div>
        </div>
        <div class="card">
            <div class="label">Final Value</div>
            <div class="value {'positive' if m.total_return >= 0 else 'negative'}">
                ${result.equity_curve.iloc[-1]:,.0f}
            </div>
        </div>
        <div class="card">
            <div class="label">Annualized Return</div>
            <div class="value {'positive' if m.annualized_return >= 0 else 'negative'}">
                {pct_fmt(m.annualized_return)}
            </div>
        </div>
    </div>

    <div class="section-title">📊 Detailed Metrics</div>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Sortino Ratio</td><td>{m.sortino_ratio:.3f}</td></tr>
        <tr><td>Calmar Ratio</td><td>{m.calmar_ratio:.3f}</td></tr>
        <tr><td>Volatility (Ann.)</td><td>{m.volatility:.2%}</td></tr>
        <tr><td>Downside Vol.</td><td>{m.downside_volatility:.2%}</td></tr>
        <tr><td>VaR (95%)</td><td>{m.var_95:.2%}</td></tr>
        <tr><td>CVaR (95%)</td><td>{m.cvar_95:.2%}</td></tr>
        <tr><td>Max DD Duration</td><td>{m.max_drawdown_duration} days</td></tr>
        <tr><td>Avg Win</td><td>{m.avg_win:.4f}</td></tr>
        <tr><td>Avg Loss</td><td>{m.avg_loss:.4f}</td></tr>
        <tr><td>Largest Win</td><td>{m.largest_win:.4f}</td></tr>
        <tr><td>Largest Loss</td><td>{m.largest_loss:.4f}</td></tr>
        <tr><td>Expectancy</td><td>{m.expectancy:.4f}</td></tr>
        <tr><td>Benchmark Return</td><td>{pct_fmt(m.benchmark_return)}</td></tr>
        <tr><td>Alpha</td><td>{m.alpha:.4f}</td></tr>
        <tr><td>Beta</td><td>{m.beta:.4f}</td></tr>
    </table>
"""

        # Add trade log
        if not result.trades.empty:
            html += """
    <div class="section-title">📋 Trade Log</div>
    <table>
        <tr>
            <th>#</th><th>Side</th><th>Entry</th><th>Exit</th>
            <th>Qty</th><th>PnL</th><th>PnL %</th><th>Entry Time</th><th>Exit Time</th>
        </tr>
"""
            for i, (_, trade) in enumerate(result.trades.iterrows()):
                pnl_cls = "positive" if trade["pnl"] > 0 else "negative"
                html += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{trade['side']}</td>
            <td>${trade['entry_price']:.2f}</td>
            <td>${trade['exit_price']:.2f}</td>
            <td>{trade['quantity']:.4f}</td>
            <td class="{pnl_cls}">${trade['pnl']:.2f}</td>
            <td class="{pnl_cls}">{trade['pnl_pct']:.2%}</td>
            <td>{trade.get('entry_time', '')}</td>
            <td>{trade.get('exit_time', '')}</td>
        </tr>
"""
            html += "    </table>\n"

        html += """
    <div style="text-align: center; margin-top: 2rem; color: #555;">
        Generated by QuantPilot AI Engine
    </div>
</body>
</html>
"""

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to {filepath}")

        return html

    @staticmethod
    def compare_backtests(
        results: list[Any],
        names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple backtest results side by side.

        Args:
            results: List of BacktestResult objects
            names: Display names for each result

        Returns:
            DataFrame comparison table
        """
        if names is None:
            names = [f"{r.symbol}" for r in results]

        comparison = {}
        for name, result in zip(names, results):
            m = result.metrics
            comparison[name] = {
                "Total Return": f"{m.total_return:.2%}",
                "Ann. Return": f"{m.annualized_return:.2%}",
                "Sharpe": f"{m.sharpe_ratio:.3f}",
                "Sortino": f"{m.sortino_ratio:.3f}",
                "Max Drawdown": f"{m.max_drawdown:.2%}",
                "Calmar": f"{m.calmar_ratio:.3f}",
                "Volatility": f"{m.volatility:.2%}",
                "Win Rate": f"{m.win_rate:.1%}",
                "Profit Factor": f"{m.profit_factor:.3f}",
                "Trades": m.total_trades,
                "Final Value": f"${result.equity_curve.iloc[-1]:,.0f}",
            }

        return pd.DataFrame(comparison)
