"""
QuantPilot AI — Custom Dark Theme CSS

Premium dark-themed styling with glassmorphism, neon accents, and micro-animations.
"""

CUSTOM_CSS = """
<style>
    /* ── Google Fonts ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ──────────────────────────────────── */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #12172b;
        --bg-tertiary: #1a1f35;
        --bg-card: rgba(18, 23, 43, 0.8);
        --bg-glass: rgba(26, 31, 53, 0.6);
        --border: rgba(42, 47, 69, 0.8);
        --border-glow: rgba(0, 212, 255, 0.2);

        --text-primary: #e4e6eb;
        --text-secondary: #8b8fa3;
        --text-muted: #555a6e;

        --accent-cyan: #00d4ff;
        --accent-purple: #7c3aed;
        --accent-magenta: #ff006e;
        --accent-green: #00ff88;
        --accent-red: #ff4444;
        --accent-orange: #ff8c00;
        --accent-gold: #ffd700;

        --gradient-primary: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        --gradient-success: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        --gradient-danger: linear-gradient(135deg, #ff4444 0%, #ff006e 100%);
        --gradient-card: linear-gradient(145deg, rgba(18,23,43,0.9) 0%, rgba(10,14,23,0.95) 100%);

        --shadow-glow: 0 0 20px rgba(0, 212, 255, 0.1);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);

        --radius: 12px;
        --radius-lg: 16px;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* ── Global Overrides ───────────────────────────────── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #0a0e17 100%) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: -0.02em;
    }

    /* ── Metric Cards ───────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--gradient-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1rem 1.2rem !important;
        box-shadow: var(--shadow-card) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--border-glow) !important;
        box-shadow: var(--shadow-glow) !important;
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        font-family: var(--font-mono) !important;
    }

    [data-testid="stMetricDelta"] > div {
        font-family: var(--font-mono) !important;
        font-weight: 600 !important;
    }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-secondary);
        border-radius: var(--radius);
        padding: 0.3rem;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(0, 212, 255, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: 0.85rem;
    }

    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4) !important;
        transform: translateY(-1px);
    }

    /* ── Selectbox & Inputs ──────────────────────────────── */
    [data-testid="stSelectbox"], .stTextInput, .stNumberInput {
        background: var(--bg-secondary) !important;
        border-radius: var(--radius) !important;
    }

    /* ── DataFrames ──────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Expander ────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* ── Custom Classes ──────────────────────────────────── */
    .kpi-card {
        background: var(--gradient-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        border-color: var(--border-glow);
        box-shadow: var(--shadow-glow);
    }

    .section-title {
        color: var(--accent-cyan);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    .profit { color: var(--accent-green) !important; }
    .loss { color: var(--accent-red) !important; }
    .neutral { color: var(--text-secondary) !important; }

    /* ── Animations ──────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stMetric, .stPlotlyChart {
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.2); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
    }

    .live-indicator {
        animation: pulse-glow 2s infinite;
    }

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
"""


def inject_css():
    """Inject custom CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
