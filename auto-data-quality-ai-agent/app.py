from __future__ import annotations

import os
import streamlit as st
from typing import List

from dotenv import load_dotenv

from config import get_settings, Settings
from supabase_client.connection import get_client
from supabase_client import metadata as sb_metadata
from ui.layout import render_header, render_sidebar
from ui import tabs as ui_tabs


st.set_page_config(page_title="AI Data Quality Assistant", layout="wide")

# Ensure .env is loaded before reading settings in cases where Streamlit is run from other dirs.
load_dotenv(override=True)


@st.cache_resource(show_spinner=False)
def _get_cached_client(supabase_url: str, supabase_key: str):
    # Cache is keyed by the credentials to avoid stale clients after .env changes.
    settings = Settings(supabase_url=supabase_url, supabase_key=supabase_key, supabase_anon_key=None)
    return get_client(settings=settings)


def _safe_list_tables(client, schema: str) -> List[str]:
    try:
        return sb_metadata.list_tables(client, schema=schema)
    except Exception:
        return []


def main():
    st.markdown(
        """
        <style>
        /* Global look inspired by the provided mock */
        body, .stApp {
            background: radial-gradient(circle at 20% 20%, #0f1f3a 0%, #0b1424 30%, #060b14 60%, #04060d 100%);
            color: #e6f0ff;
        }
        .block-container { padding-top: 1rem; }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1f3a 0%, #0c162b 40%, #0b0f1f 100%);
            color: #e6f0ff;
            border-right: 1px solid rgba(255,255,255,0.08);
            box-shadow: 4px 0 20px rgba(0,0,0,0.35);
        }
        /* Cards */
        .dq-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        }
        .dq-card h4 { margin: 0 0 0.35rem 0; color: #e6f5ff; }
        .dq-card .dq-pill {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-right: 0.35rem;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.15);
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #2cb2ff, #0ae3ff);
            color: #0b1226;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(12, 194, 255, 0.35);
        }
        .stButton>button:hover { filter: brightness(1.05); }
        /* Tabs */
        .stTabs [role=\"tablist\"] button {
            background: rgba(255,255,255,0.08);
            border-radius: 12px 12px 0 0;
            color: #e6f0ff;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .stTabs [role=\"tablist\"] button[aria-selected=\"true\"] {
            background: linear-gradient(90deg, #1f3b63, #18243a);
            border-bottom: 2px solid #0ae3ff;
        }
        /* Table headers */
        .dataframe thead tr th {
            background: rgba(255,255,255,0.08) !important;
            color: #e6f0ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_header()

    # Session defaults
    st.session_state.setdefault("profiling", None)
    st.session_state.setdefault("issues", [])
    st.session_state.setdefault("ai_analysis", None)
    st.session_state.setdefault("latest_run_id", None)
    st.session_state["openai_key_present"] = bool(
        st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else False
        or os.environ.get("OPENAI_API_KEY")
    )
    st.session_state.setdefault("selected_schema", "public")
    st.session_state.setdefault("selected_table", None)

    connection_ok = False
    client = None
    connection_error = None
    settings = None
    key_hint = None
    try:
        # Validate env early
        settings = get_settings()
        key_hint = f"{settings.supabase_key[:6]}... (len={len(settings.supabase_key)})"
        client = _get_cached_client(settings.supabase_url, settings.supabase_key)
        connection_ok = True
    except Exception as exc:
        connection_error = str(exc)
        st.warning(f"Supabase connection not ready: {exc}")

    # Build sidebar selections
    current_schema = st.session_state.get("selected_schema", "public")
    tables = _safe_list_tables(client, current_schema) if connection_ok else []

    selected_schema, selected_table = render_sidebar(
        connection_ok=connection_ok,
        tables=tables,
        default_schema=current_schema,
        connection_error=connection_error,
        key_hint=key_hint if not connection_ok else None,
    )
    st.session_state["selected_schema"] = selected_schema
    st.session_state["selected_table"] = selected_table
    st.session_state["sb_client"] = client

    tab_connect, tab_profile, tab_ai, tab_history = st.tabs(
        ["Connect & Run Checks", "Profile & Issues", "AI Analysis & Fix Suggestions", "Run History"]
    )

    with tab_connect:
        ui_tabs.render_connect_and_run_tab(client, selected_schema, selected_table)
    with tab_profile:
        ui_tabs.render_profile_and_issues_tab()
    with tab_ai:
        ui_tabs.render_ai_analysis_tab()
    with tab_history:
        ui_tabs.render_run_history_tab()

    st.markdown("---")
    st.caption("The AI agent operates only on aggregated profiling metadata and issue summaries. Raw data never leaves the app.")


if __name__ == "__main__":
    main()
