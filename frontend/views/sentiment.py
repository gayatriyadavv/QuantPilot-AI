"""
QuantPilot AI — Sentiment Analysis Page

Financial news sentiment analysis using FinBERT.
"""

from __future__ import annotations

import asyncio
from frontend.utils import run_async
from datetime import datetime

import streamlit as st

from frontend.components.charts import create_sentiment_gauge
from frontend.components.metrics import metric_row, render_kpi_card


def render_sentiment():
    """Render the sentiment analysis page."""
    st.markdown("## 📰 Sentiment Analysis")
    st.markdown("Powered by **FinBERT** (ProsusAI/finbert) — Financial domain NLP")

    # ── Symbol Sentiment ──────────────────────────────────────
    st.markdown("### 🔍 Analyze Symbol Sentiment")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"], key="sent_sym")
    with col2:
        max_headlines = st.slider("Headlines", 5, 20, 10, key="sent_limit")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔬 Analyze", use_container_width=True, type="primary", key="sent_analyze")

    if analyze_btn:
        with st.spinner(f"Analyzing sentiment for {symbol}..."):
            try:
                from backend.sentiment.news_fetcher import NewsFetcher
                from backend.sentiment.analyzer import SentimentAnalyzer

                # Fetch news
                fetcher = NewsFetcher()
                headlines = run_async(fetcher.fetch_yahoo_news(symbol, limit=max_headlines))

                if not headlines:
                    st.warning(f"No recent news found for {symbol}")
                    return

                # Analyze sentiment
                analyzer = SentimentAnalyzer()
                texts = [h["title"] for h in headlines]
                results = analyzer.analyze_batch(texts)

                st.session_state["sent_results"] = {
                    "symbol": symbol,
                    "headlines": headlines,
                    "results": results,
                }
                st.success(f"✅ Analyzed {len(results)} headlines for {symbol}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # ── Display Results ───────────────────────────────────────
    sent_data = st.session_state.get("sent_results")

    if sent_data:
        results = sent_data["results"]
        headlines = sent_data["headlines"]

        # Calculate overall sentiment
        avg_pos = sum(r["positive"] for r in results) / len(results)
        avg_neg = sum(r["negative"] for r in results) / len(results)
        avg_neu = sum(r["neutral"] for r in results) / len(results)
        overall_score = avg_pos - avg_neg

        # Sentiment Gauge + KPIs
        col_gauge, col_kpis = st.columns([1, 2])

        with col_gauge:
            fig_gauge = create_sentiment_gauge(overall_score, f"{sent_data['symbol']} Sentiment")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_kpis:
            overall_label = "🟢 Bullish" if overall_score > 0.1 else "🔴 Bearish" if overall_score < -0.1 else "🟡 Neutral"
            metric_row([
                {"label": "Overall", "value": overall_label},
                {"label": "Positive", "value": f"{avg_pos:.1%}"},
                {"label": "Negative", "value": f"{avg_neg:.1%}"},
                {"label": "Neutral", "value": f"{avg_neu:.1%}"},
            ])

            bullish_count = sum(1 for r in results if r["label"] == "positive")
            bearish_count = sum(1 for r in results if r["label"] == "negative")
            neutral_count = sum(1 for r in results if r["label"] == "neutral")

            metric_row([
                {"label": "Headlines", "value": str(len(results))},
                {"label": "Bullish", "value": str(bullish_count)},
                {"label": "Bearish", "value": str(bearish_count)},
                {"label": "Neutral", "value": str(neutral_count)},
            ])

        # Headline Details
        st.markdown("---")
        st.markdown("### 📋 Headline Breakdown")

        for i, (headline, result) in enumerate(zip(headlines, results)):
            label = result["label"]
            confidence = result["confidence"]
            icon = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
            bar_color = "#00ff88" if label == "positive" else "#ff4444" if label == "negative" else "#8b8fa3"

            st.markdown(f"""
            <div style="
                background: linear-gradient(145deg, rgba(18,23,43,0.9), rgba(10,14,23,0.95));
                border: 1px solid rgba(42,47,69,0.8);
                border-left: 3px solid {bar_color};
                border-radius: 8px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.5rem;
            ">
                <div style="color: #e4e6eb; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    {icon} {headline['title']}
                </div>
                <div style="color: #8b8fa3; font-size: 0.75rem;">
                    {label.upper()} ({confidence:.0%} confidence) |
                    +{result['positive']:.2f} / -{result['negative']:.2f} / ○{result['neutral']:.2f}
                    {f" | {headline.get('source', '')}" if headline.get('source') else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Custom Text Analysis ──────────────────────────────────
    st.markdown("---")
    st.markdown("### ✍️ Custom Text Analysis")

    custom_text = st.text_area(
        "Enter financial text to analyze:",
        placeholder="e.g., 'Apple reports record quarterly earnings, beating analyst expectations'",
        height=100,
        key="sent_custom",
    )

    if st.button("🔬 Analyze Text", key="sent_custom_btn") and custom_text:
        with st.spinner("Analyzing..."):
            try:
                from backend.sentiment.analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                result = analyzer.analyze_text(custom_text)

                label = result["label"]
                icon = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"

                st.markdown(f"""
                ### Result: {icon} {label.upper()}

                | Metric | Score |
                |--------|-------|
                | Positive | {result['positive']:.4f} |
                | Negative | {result['negative']:.4f} |
                | Neutral | {result['neutral']:.4f} |
                | Confidence | {result['confidence']:.2%} |
                """)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
