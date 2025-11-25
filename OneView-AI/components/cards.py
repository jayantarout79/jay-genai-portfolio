"""Reusable card components and helper renderers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import streamlit as st

from theme.style import sentiment_color


def metric_pill(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-pill'><span class='label'>{label}</span><span class='value'>{value}</span></div>",
        unsafe_allow_html=True,
    )


def timeline_card(timestamp: str, title: str, description: str, sentiment: str, tags: Sequence[str] | None = None):
    color = sentiment_color(sentiment)
    tag_str = " ".join(f"<span style='background:#eef2ff;border-radius:12px;padding:2px 10px;margin-right:6px;font-size:0.75rem;color:#3f4c5d;'>{tag}</span>" for tag in tags or [])
    st.markdown(
        f"""
        <div class='card' style='margin-bottom:0.7rem; border-left:4px solid {color};'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <strong style='font-size:0.95rem;'>{title}</strong>
                <span style='color:{color};font-weight:600;'>{timestamp}</span>
            </div>
            <p style='color:#4f5d75;margin-top:0.35rem;'>{description}</p>
            <div style='margin-top:0.5rem;'>{tag_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_list(title: str, items: Iterable[str], color: str = "#2563eb"):
    st.markdown(f"<h4 style='margin-bottom:0.4rem;'>{title}</h4>", unsafe_allow_html=True)
    for item in items:
        st.markdown(
            f"<div style='border-left:3px solid {color};padding-left:0.6rem;margin-bottom:0.4rem;color:#3a475c;'>{item}</div>",
            unsafe_allow_html=True,
        )


def chat_bubble(role: str, content: str, timestamp: str | None = None):
    bubble_class = "assistant" if role == "assistant" else "user"
    ts_html = f"<div style='font-size:0.75rem;color:#95a0b5;margin-bottom:0.2rem;'>{timestamp}</div>" if timestamp else ""
    st.markdown(
        f"<div class='chat-bubble {bubble_class}'>{ts_html}{content}</div>",
        unsafe_allow_html=True,
    )

