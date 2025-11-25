"""Plotly visualization builders for Streamlit rendering."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


COLOR_MAP = {"positive": "#14b8a6", "neutral": "#f5a524", "negative": "#ef4444"}


def _base_layout(title: str) -> dict:
    return dict(
        template="simple_white",
        title=title,
        margin=dict(l=40, r=20, t=40, b=35),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )


def sentiment_timeline(segment_df: pd.DataFrame) -> go.Figure:
    """Create a sentiment timeline scatter/line chart."""
    fig = go.Figure()
    if segment_df is None or segment_df.empty:
        fig.update_layout(**_base_layout("Sentiment Timeline"))
        return fig

    times = segment_df["start"] / 60.0
    sentiments = segment_df["sentiment_score"]
    hover_text = [
        f"{row.segment_type.title()} | {', '.join(row.topics) if row.topics else 'Topics TBD'}<br>{row.snippet}"
        for row in segment_df.itertuples()
    ]
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sentiments,
            mode="lines+markers",
            marker=dict(
                size=9,
                color=[COLOR_MAP.get(label, "#2563eb") for label in segment_df["sentiment_label"]],
            ),
            line=dict(color="#94a3b8", width=2),
            hovertext=hover_text,
            hoverinfo="text+x+y",
            name="Sentiment",
        )
    )
    fig.update_layout(
        **_base_layout("Sentiment Timeline"),
        xaxis_title="Time (minutes)",
        yaxis_title="Sentiment Score (-1 → 1)",
        yaxis=dict(range=[-1.05, 1.05]),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, bgcolor="#eef2ff", thickness=0.08))
    return fig


def sentiment_distribution(segment_df: pd.DataFrame) -> go.Figure:
    """Build a sentiment distribution bar chart."""
    fig = go.Figure()
    if segment_df is None or segment_df.empty:
        fig.update_layout(**_base_layout("Sentiment Distribution"))
        return fig

    counts = segment_df["sentiment_label"].value_counts()
    fig.add_trace(
        go.Bar(
            x=[label.title() for label in counts.index],
            y=counts.values,
            marker_color=[COLOR_MAP.get(label, "#94a3b8") for label in counts.index],
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout("Sentiment Distribution"),
        xaxis_title="Sentiment",
        yaxis_title="Segments",
    )
    return fig


def topic_frequency(topic_counts: Optional[dict]) -> go.Figure:
    """Render topic frequency as a bar chart."""
    fig = go.Figure()
    if not topic_counts:
        fig.update_layout(**_base_layout("Topic Frequency"))
        return fig

    labels = list(topic_counts.keys())
    values = list(topic_counts.values())
    fig.add_trace(
        go.Bar(
            x=[label.title() for label in labels],
            y=values,
            marker_color="#2563eb",
        )
    )
    fig.update_layout(
        **_base_layout("Topic Frequency"),
        xaxis_title="Topic",
        yaxis_title="Mentions",
    )
    return fig


def energy_intensity(segment_df: pd.DataFrame) -> go.Figure:
    """Plot the absolute sentiment intensity over time."""
    fig = go.Figure()
    if segment_df is None or segment_df.empty:
        fig.update_layout(**_base_layout("Energy / Intensity"))
        return fig

    fig.add_trace(
        go.Scatter(
            x=segment_df["start"] / 60.0,
            y=segment_df["sentiment_score"].abs(),
            fill="tozeroy",
            line=dict(color="#7c3aed"),
            name="Intensity",
        )
    )
    fig.update_layout(
        **_base_layout("Energy / Intensity"),
        xaxis_title="Time (minutes)",
        yaxis_title="Intensity",
        yaxis=dict(range=[0, 1]),
    )
    return fig


def keyword_heatmap(segment_df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Show a heatmap of keyword mentions across segments."""
    fig = go.Figure()
    if segment_df is None or segment_df.empty:
        fig.update_layout(**_base_layout("Keyword Evolution"))
        return fig

    topic_counts = {}
    topic_presence = {}
    for idx, row in enumerate(segment_df.itertuples()):
        topics = row.topics or []
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            topic_presence.setdefault(topic, []).append((idx, 1))

    if not topic_counts:
        fig.update_layout(**_base_layout("Keyword Evolution"))
        return fig

    top_topics = [topic for topic, _ in sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    matrix = []
    for topic in top_topics:
        row = [0.0] * len(segment_df)
        for idx, _ in topic_presence.get(topic, []):
            row[idx] = 1.0
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"{int(s/60)}:{int(s%60):02d}" for s in segment_df["start"]],
            y=[topic.title() for topic in top_topics],
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(**_base_layout("Keyword Evolution"), xaxis_title="Timeline", yaxis_title="Keyword")
    return fig


def segment_type_share(segment_df: pd.DataFrame) -> go.Figure:
    """Pie chart of segment_type mix as a proxy for speaker/activity dominance."""
    fig = go.Figure()
    if segment_df is None or segment_df.empty:
        fig.update_layout(**_base_layout("Conversation Mix"))
        return fig

    counts = segment_df["segment_type"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=[label.title() for label in counts.index],
            values=counts.values,
            hole=0.4,
        )
    )
    fig.update_layout(**_base_layout("Conversation Mix"), showlegend=True)
    return fig
