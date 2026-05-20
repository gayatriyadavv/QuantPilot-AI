"""
QuantPilot AI — Main Streamlit Application

Premium dark-themed trading dashboard with multi-page navigation.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page Config (MUST be first Streamlit call) ───────────────

st.set_page_config(
    page_title="QuantPilot AI — Trading Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "QuantPilot AI — AI-Powered Trading Platform",
    },
)

# ── Inject Custom Theme ──────────────────────────────────────

from frontend.styles import inject_css
inject_css()

# ── Import Pages ──────────────────────────────────────────────

from frontend.views.dashboard import render_dashboard
from frontend.views.backtesting import render_backtesting
from frontend.views.portfolio import render_portfolio
from frontend.views.agents import render_agents
from frontend.views.sentiment import render_sentiment

# ── Sidebar Navigation ───────────────────────────────────────

with st.sidebar:
    st.markdown("# 🚀 QuantPilot AI")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        options=[
            "📈 Dashboard",
            "🔬 Backtesting",
            "💼 Portfolio",
            "🤖 AI Agents",
            "📰 Sentiment",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick Settings
    st.markdown("### ⚙️ Quick Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000", key="api_url")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="auto_refresh")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #555a6e; font-size: 0.75rem;">
            <div>QuantPilot AI v1.0.0</div>
            <div>Powered by RL + PyTorch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Page Routing ──────────────────────────────────────────────

if page == "📈 Dashboard":
    render_dashboard()
elif page == "🔬 Backtesting":
    render_backtesting()
elif page == "💼 Portfolio":
    render_portfolio()
elif page == "🤖 AI Agents":
    render_agents()
elif page == "📰 Sentiment":
    render_sentiment()
