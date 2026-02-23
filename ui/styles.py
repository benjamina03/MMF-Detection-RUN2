import streamlit as st


def load_custom_css():
    st.markdown(
        """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff !important;
        color: #111111 !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.92) !important;
        border-bottom: 1px solid #dbeafd;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    .stMarkdown, .stText, p, label, span, div {
        color: #111111;
    }

    .stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label, .stFileUploader label {
        color: #111111 !important;
    }

    /* Main container - Light theme */
    .main {
        background: radial-gradient(circle at top right, #eaf4ff 0%, #f8fbff 45%, #ffffff 100%);
    }

    /* Metric cards - White with blue accents */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px solid #3498db;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.1);
    }

    div[data-testid="metric-container"] > label {
        color: #2c3e50;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="metric-container"] > div {
        font-size: 32px;
        font-weight: 700;
        color: #2980b9;
    }

    .dataframe {
        font-size: 14px;
        background-color: white;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #ffffff !important;
        color: #111111 !important;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #3498db;
        color: white;
        border: none;
        transition: all 0.3s ease;
        min-height: 40px;
    }

    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
        margin-bottom: 20px;
        border: 1px solid #e3f2fd;
    }

    .objective-card {
        background: linear-gradient(180deg, #ffffff 0%, #f3f9ff 100%);
        border: 1px solid #d7e9fb;
        border-left: 5px solid #2d9bf0;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .objective-complete {
        border-left-color: #17a673;
    }

    .login-shell {
        background: linear-gradient(145deg, #eef6ff 0%, #ffffff 65%);
        border: 1px solid #d6e9fb;
        border-radius: 20px;
        padding: 30px;
        margin-top: 10px;
        box-shadow: 0 16px 34px rgba(45, 155, 240, 0.14);
    }

    .login-title {
        color: #1f5f99;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .login-subtitle {
        color: #5d7f9f;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .login-note {
        background: #f3f9ff;
        border-left: 4px solid #2d9bf0;
        border-radius: 8px;
        padding: 12px;
        color: #376185;
        font-size: 13px;
        margin-top: 14px;
    }

    .login-badge {
        display: inline-block;
        background: #e7f3ff;
        color: #1f5f99;
        border: 1px solid #cbe4ff;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 8px;
    }

    .login-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin-top: 10px;
    }

    .login-feature-item {
        background: #f4f9ff;
        border: 1px solid #dcecff;
        border-radius: 10px;
        padding: 10px 8px;
        text-align: center;
        color: #2d5f8d;
        font-size: 12px;
        font-weight: 700;
    }

    .login-form-title {
        color: #1f5f99;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .login-form-subtitle {
        color: #6d8ca8;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .hero-panel {
        background: linear-gradient(110deg, #1f5f99 0%, #2d9bf0 60%, #78bfff 100%);
        border-radius: 16px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(31, 95, 153, 0.25);
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 14px;
        opacity: 0.95;
        margin-bottom: 10px;
    }

    .soft-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
    }

    .landing-hero {
        background: linear-gradient(118deg, #0f3d63 0%, #1f5f99 55%, #49a6ff 100%);
        border-radius: 18px;
        padding: 28px;
        color: #ffffff;
        box-shadow: 0 14px 30px rgba(16, 65, 106, 0.28);
        margin-bottom: 18px;
    }

    .landing-hero h1 {
        margin: 0 0 8px 0;
        font-size: 34px;
        line-height: 1.15;
        font-weight: 800;
    }

    .landing-hero p {
        margin: 0 0 14px 0;
        font-size: 15px;
        opacity: 0.95;
    }

    .landing-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.36);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .landing-card {
        background: #ffffff;
        border: 1px solid #d9ebfc;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 18px rgba(45, 155, 240, 0.09);
        margin-bottom: 12px;
    }

    .landing-card h4 {
        color: #1f5f99;
        margin: 0 0 10px 0;
    }

    .landing-card p, .landing-card li {
        color: #4f7090;
        font-size: 14px;
    }

    .landing-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 14px;
    }

    .landing-stat-item {
        background: #ffffff;
        border: 1px solid #d9ebfc;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 6px 14px rgba(45, 155, 240, 0.08);
    }

    .landing-stat-value {
        color: #1f5f99;
        font-size: 22px;
        font-weight: 800;
        line-height: 1.1;
    }

    .landing-stat-label {
        color: #6a89a5;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }

    .landing-flow {
        background: #ffffff;
        border: 1px solid #d9ebfc;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 18px rgba(45, 155, 240, 0.09);
        margin-bottom: 12px;
    }

    .landing-flow h4 {
        color: #1f5f99;
        margin: 0 0 10px 0;
    }

    .landing-flow-step {
        background: #f4f9ff;
        border: 1px solid #dcecff;
        border-radius: 10px;
        padding: 9px 10px;
        color: #42688c;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .transaction-card {
        background: white;
        border: 1px solid #e3f2fd;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.08);
        transition: all 0.2s ease;
    }

    .transaction-card:hover {
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
        transform: translateY(-2px);
    }

    .badge-clean {
        background-color: #27ae60;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-fraud {
        background-color: #e74c3c;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-critical {
        background-color: #e74c3c;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-high {
        background-color: #e67e22;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-medium {
        background-color: #f39c12;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-low {
        background-color: #27ae60;
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 18px;
        margin-top: 35px;
        border-bottom: 3px solid #3498db;
        padding-bottom: 8px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dbeafd;
    }

    section[data-testid="stSidebar"] * {
        color: #1f2d3d;
    }

    .sidebar-brand-card {
        background: linear-gradient(130deg, #f5faff 0%, #edf6ff 100%);
        border: 1px solid #d7e9fb;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .sidebar-brand-title {
        font-size: 16px;
        font-weight: 800;
        color: #1f5f99;
        margin-bottom: 4px;
    }

    .sidebar-brand-sub {
        font-size: 12px;
        color: #4f7090;
        margin-bottom: 0;
    }

    .sidebar-user-card {
        background: #f6fbff;
        border: 1px solid #d7e9fb;
        border-radius: 12px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    .sidebar-user-title {
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #1f5f99;
    }

    .sidebar-user-line {
        font-size: 13px;
        color: #4f7090;
        margin-bottom: 4px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
