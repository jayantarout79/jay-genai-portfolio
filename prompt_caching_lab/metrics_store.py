"""
Simple in-memory metrics store that survives across Streamlit reruns via session state.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st


def init_store() -> None:
    """Initialize the runs list in session_state."""
    if "runs" not in st.session_state:
        st.session_state["runs"]: List[Dict] = []


def add_run(record: Dict) -> None:
    """Append a run record."""
    st.session_state["runs"].append(record)


def get_runs_df() -> pd.DataFrame:
    """Return a DataFrame of runs, or an empty DataFrame if none exist."""
    runs = st.session_state.get("runs", [])
    if not runs:
        return pd.DataFrame()
    return pd.DataFrame(runs)
