"""Streamlit entry point for the Customer Experience Analyzer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai.llm_client import LLMClient
from core.config import load_config
from core.data_loader import load_survey_data
from ui import state
from ui.layout import (
    render_chat_tab,
    render_comparison_tab,
    render_nps_tab,
    render_summary_tab,
    render_themes_tab,
    render_upload_tab,
)

DATA_PATH = Path("data/sample_survey.csv")


@st.cache_data(show_spinner=False)
def load_demo_dataset() -> pd.DataFrame:
    return load_survey_data(DATA_PATH)


def build_llm_client(override_key: str | None) -> LLMClient:
    config = load_config(api_key_override=override_key)
    return LLMClient(config.openai_api_key, config.openai_model)


def _inject_dashboard_theme() -> None:
    """Apply global CSS to give the app a polished dashboard feel."""

    st.markdown(
        """
        <style>
            :root {
                --accent-color: #7c5dff;
                --accent-color-light: #a088ff;
                --card-bg: rgba(17, 20, 40, 0.9);
                --border-color: rgba(255,255,255,0.08);
            }
            .stApp {
                background: radial-gradient(circle at 10% 20%, rgba(74,76,120,0.4), transparent 60%),
                            radial-gradient(circle at 80% 0%, rgba(110,83,255,0.3), transparent 55%),
                            #05060d;
                color: #f5f7ff;
            }
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            section[data-testid="stSidebar"] > div:first-child {
                background: #0f1122;
                border-right: 1px solid rgba(255,255,255,0.05);
            }
            .section-card {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 18px;
                padding: 1.3rem 1.6rem;
                margin-bottom: 1rem;
                box-shadow: 0px 10px 30px rgba(0,0,0,0.35);
            }
            .metric-card div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.04);
                border-radius: 16px;
                padding: 1rem;
                border: 1px solid rgba(255,255,255,0.06);
            }
            .kpi-card {
                background: rgba(255,255,255,0.04);
                border-radius: 14px;
                padding: 0.8rem 1rem;
                border: 1px solid rgba(255,255,255,0.08);
                cursor: help;
            }
            .kpi-label {
                font-size: 0.8rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(255,255,255,0.6);
            }
            .kpi-value {
                font-size: 1.4rem;
                font-weight: 700;
            }
            div[data-baseweb="select"] > div {
                border-radius: 12px;
                border-color: rgba(255,255,255,0.2);
                background: rgba(255,255,255,0.02);
            }
            .sidebar-card {
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(255,255,255,0.07);
            }
            .sidebar-card.subtle {
                background: rgba(255,255,255,0.03);
            }
            .sidebar-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }
            .sidebar-label {
                display: block;
                font-size: 0.85rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(255,255,255,0.6);
                margin-bottom: 0.3rem;
            }
            .pill {
                background: rgba(255,255,255,0.08);
                border-radius: 999px;
                padding: 0.3rem 0.75rem;
                font-size: 0.85rem;
                display: inline-flex;
            }
            .sidebar-input input {
                border-radius: 14px !important;
                border: 1px solid rgba(255,255,255,0.45) !important;
                background: rgba(7,8,16,0.8) !important;
                color: #fff !important;
                padding: 0.4rem 0.8rem !important;
            }
            .filter-card {
                background: rgba(14,18,34,0.92);
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.04);
                padding: 1.2rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: inset 0 0 20px rgba(79,57,255,0.18);
            }
            .filter-card-title {
                font-weight: 600;
                margin-bottom: 0.8rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                font-size: 0.8rem;
                color: rgba(255,255,255,0.7);
            }
            button[kind="primary"] {
                background: linear-gradient(135deg, var(--accent-color), var(--accent-color-light));
                border: none;
                color: white;
                font-weight: 600;
                box-shadow: 0px 12px 24px rgba(108,83,255,0.35);
            }
            button[kind="primary"]:hover {
                background: linear-gradient(135deg, var(--accent-color-light), var(--accent-color));
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                border-bottom: 1px solid rgba(255,255,255,0.08);
            }
            .stTabs [data-baseweb="tab"] {
                color: rgba(255,255,255,0.85);
                font-weight: 700;
                font-size: 0.95rem;
                padding: 0.6rem 1rem;
                border-radius: 12px 12px 0 0;
                background: rgba(255,255,255,0.04);
                margin-bottom: -1px;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(124,93,255,0.18) !important;
                color: #fff !important;
            }
            .keyword-cloud-card {
                background: rgba(18,20,35,0.95);
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.06);
                box-shadow: inset 0 0 20px rgba(91,63,255,0.15);
            }
            .keyword-cloud-header {
                font-weight: 600;
                margin-bottom: 0.6rem;
            }
            .keyword-cloud-words {
                display: flex;
                flex-wrap: wrap;
                gap: 0.8rem;
                justify-content: center;
                align-items: center;
                text-transform: lowercase;
            }
            .theme-card {
                background: rgba(255,255,255,0.03);
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.08);
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            }
            .theme-card-header {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
            }
            .theme-name {
                font-weight: 600;
                font-size: 1.1rem;
            }
            .theme-description {
                color: rgba(255,255,255,0.7);
                font-size: 0.9rem;
            }
            .theme-badge {
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.75rem;
            }
            .theme-badge.sentiment-positive { background: rgba(61,201,142,0.3); color:#76ffbf; }
            .theme-badge.sentiment-neutral { background: rgba(255,255,255,0.15); color:#fff; }
            .theme-badge.sentiment-negative { background: rgba(255,112,112,0.25); color:#ff8f8f; }
            .theme-metrics {
                display: flex;
                gap: 1rem;
                margin-top: 0.8rem;
            }
            .theme-metrics div span {
                display: block;
                font-size: 0.8rem;
                color: rgba(255,255,255,0.6);
            }
            .theme-comments ul {
                padding-left: 1rem;
                margin: 0.4rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="CX Analyzer", layout="wide")
    _inject_dashboard_theme()
    state.init_session_state()

    # Sidebar controls ------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">CX Analyzer</div>
                <p>Upload survey data, analyze NPS, and ask AI for insight.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        previous_key = st.session_state.get("api_key_override", "")
        st.markdown('<label class="sidebar-label">OpenAI API Key</label>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-input">', unsafe_allow_html=True)
        api_key = st.text_input(
            "OpenAI API Key",
            value=previous_key,
            type="password",
            label_visibility="collapsed",
            help="Optional override for this session.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if api_key != previous_key:
            st.session_state["api_key_override"] = api_key
            st.session_state["api_key_saved_notice"] = bool(api_key)
        if st.session_state.get("api_key_saved_notice") and api_key:
            st.success("API key saved for this session.", icon="🔐")
        st.markdown(
            f"""
            <div class="sidebar-card subtle">
                <span class="sidebar-label">Data Source</span>
                <div class="pill">{st.session_state.get("data_source", "Not loaded")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    llm_client = build_llm_client(st.session_state.get("api_key_override") or None)
    df = state.get_dataframe()

    tabs = st.tabs(
        [
            "Upload & Health",
            "NPS Overview",
            "AI Themes",
            "What Changed?",
            "Exec Summary",
            "Chatbot",
        ]
    )

    with tabs[0]:
        render_upload_tab(df, state.set_dataframe, load_demo_dataset)
        df = state.get_dataframe()

    filtered_df = pd.DataFrame()
    metrics_meta: dict[str, object] = {}
    with tabs[1]:
        filtered_df, metrics_meta = render_nps_tab(df)

    themes_selection: list[dict[str, object]] = []
    with tabs[2]:
        themes_selection = render_themes_tab(df if df is not None else pd.DataFrame(), llm_client)

    comparison_info: dict[str, object] = {}
    with tabs[3]:
        comparison_info = render_comparison_tab(df if df is not None else pd.DataFrame())

    with tabs[4]:
        summary_themes = render_summary_tab(df if df is not None else pd.DataFrame(), llm_client, comparison_info)

    with tabs[5]:
        render_chat_tab(
            df if df is not None else pd.DataFrame(),
            metrics_meta.get("metrics") if metrics_meta else None,
            summary_themes or themes_selection,
            llm_client,
            state.get_chat_history(),
            state.add_chat_message,
        )

    st.markdown(
        """
        <div class="app-hero section-card" style="margin-top:2rem;">
            <p style="margin:0 0 0.8rem 0;color:rgba(255,255,255,0.8);">
                Track health, spot emerging themes, compare cohorts, and brief leadership with data-aware AI.
            </p>
            <div class="hero-pill">Streamlit · pandas · Plotly · OpenAI · LangChain</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
