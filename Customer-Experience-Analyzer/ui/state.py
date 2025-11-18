"""Session state helpers for Streamlit."""
from __future__ import annotations

from typing import Any

import streamlit as st


def init_session_state() -> None:
    """Ensure required keys exist in st.session_state."""

    defaults = {
        "dataframe": None,
        "data_source": "",
        "chat_history": [],
        "api_key_override": "",
        "theme_cache": {},
        "metrics_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_dataframe(df, source: str) -> None:
    st.session_state["dataframe"] = df
    st.session_state["data_source"] = source
    st.session_state["metrics_cache"] = {}
    st.session_state["theme_cache"] = {}


def get_dataframe():
    return st.session_state.get("dataframe")


def get_chat_history() -> list[dict[str, str]]:
    return st.session_state.get("chat_history", [])


def add_chat_message(role: str, content: str) -> None:
    history = st.session_state.get("chat_history", [])
    history.append({"role": role, "content": content})
    st.session_state["chat_history"] = history


def reset_chat() -> None:
    st.session_state["chat_history"] = []
