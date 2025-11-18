"""Plotly visualization helpers."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def nps_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Return a line chart for the NPS trend over time."""

    if trend_df.empty:
        return go.Figure()
    fig = px.line(trend_df, x="date", y="nps", markers=True, title="NPS Trend")
    fig.update_layout(yaxis_title="NPS", xaxis_title="Date")
    peak_idx = trend_df["nps"].idxmax()
    peak_point = trend_df.loc[peak_idx]
    fig.add_trace(
        go.Scatter(
            x=[peak_point["date"]],
            y=[peak_point["nps"]],
            mode="markers+text",
            name="Peak NPS",
            text=[f"{peak_point['nps']:.2f}"],
            textposition="top center",
            marker=dict(size=12, color="#a088ff", symbol="star"),
        )
    )
    return fig


def dimension_bar_chart(df: pd.DataFrame, dimension: str) -> go.Figure:
    """Return a bar chart for NPS by a categorical dimension."""

    if df.empty:
        return go.Figure()
    fig = px.bar(
        df,
        x=dimension,
        y="nps",
        text="responses",
        title=f"NPS by {dimension.title()}",
        color="nps",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis_title="NPS",
        xaxis_title=dimension.title(),
        margin=dict(t=80, l=40, r=40, b=60),
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )
    fig.update_traces(
        texttemplate="%{text} responses",
        textposition="outside",
        cliponaxis=False,
    )
    return fig


def comparison_bar_chart(comparison_df: pd.DataFrame, dimension: str) -> go.Figure:
    """Return a grouped bar chart comparing two cohorts by dimension."""

    if comparison_df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_bar(
        x=comparison_df[dimension],
        y=comparison_df["nps_a"],
        name="Period A",
    )
    fig.add_bar(
        x=comparison_df[dimension],
        y=comparison_df["nps_b"],
        name="Period B",
    )
    fig.update_layout(
        barmode="group",
        title=f"NPS Comparison by {dimension.title()}",
        xaxis_title=dimension.title(),
        yaxis_title="NPS",
        margin=dict(t=80, l=40, r=40, b=60),
    )
    fig.update_traces(cliponaxis=False)
    return fig
