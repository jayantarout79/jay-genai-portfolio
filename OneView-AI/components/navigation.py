"""Vertical navigation component for switching dashboard sections."""

from __future__ import annotations

from typing import List

import streamlit as st


def vertical_nav(options: List[str], default: str) -> str:
    """Render a stylised vertical nav (radio list) and return the selection."""
    if default not in options:
        default = options[0]
    current = st.session_state.get("active_nav", default)
    with st.container():
        st.markdown("<div class='nav-card'>", unsafe_allow_html=True)
        selected = st.radio(
            "NAVIGATE",
            options,
            index=options.index(current),
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["active_nav"] = selected
    return selected
