from __future__ import annotations

import streamlit as st
from typing import List, Optional, Tuple


def render_header():
    st.title("AI Data Quality Assistant")
    st.caption("Supabase metadata-driven data quality analysis with safe AI assistance.")


def render_sidebar(
    connection_ok: bool,
    tables: List[str],
    default_schema: str = "public",
    connection_error: Optional[str] = None,
    key_hint: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    st.sidebar.header("Connection & Selection")
    if connection_ok:
        st.sidebar.success("Connected to Supabase")
    else:
        st.sidebar.error("Supabase connection not configured")
        if connection_error:
            st.sidebar.caption(f"Details: {connection_error}")
            if key_hint:
                st.sidebar.caption(f"Loaded key: {key_hint}")
            st.sidebar.info(
                "Use the Service Role key from Supabase Settings → API. "
                "Check for trailing spaces or quotes in .env."
            )

    schema = st.sidebar.text_input("Schema", value=default_schema)
    selected_table = None
    if tables:
        selected_table = st.sidebar.selectbox("Table", tables)
    else:
        selected_table = st.sidebar.text_input("Table name")

    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show connection help"):
        st.sidebar.info(
            "Set SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY/NEXT_PUBLIC_SUPABASE_ANON_KEY."
        )
    if not st.session_state.get("openai_key_present"):
        st.sidebar.warning("OPENAI_API_KEY not set. AI analysis will be skipped.")

    return schema or None, selected_table or None
