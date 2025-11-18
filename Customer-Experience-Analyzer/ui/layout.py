"""Reusable layout functions for the Streamlit tabs."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Iterable

import pandas as pd
import streamlit as st

from ai.chatbot import answer_question
from ai.llm_client import LLMClient
from ai.summary_generator import generate_executive_summary
from ai.theme_extractor import extract_themes
from core.data_loader import (
    DataValidationError,
    data_health,
    dataframe_profile,
    prepare_survey_dataframe,
)
from core.metrics import (
    NPSKPI,
    comparison_nps,
    comparison_table,
    compute_kpis,
    nps_by_dimension,
    nps_trend,
    subset_for_filters,
    top_keywords,
)
from core.visuals import comparison_bar_chart, dimension_bar_chart, nps_trend_chart

TIME_PRESETS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 365 days": 365,
    "Custom range": None,
}


# ---------------------------------------------------------------------------
# Tab 1
# ---------------------------------------------------------------------------

def render_upload_tab(
    df: pd.DataFrame | None,
    set_dataframe: Callable[[pd.DataFrame, str], None],
    demo_loader: Callable[[], pd.DataFrame],
) -> None:
    """Upload tab with profile + health stats."""

    st.subheader("Upload Survey Data")
    with st.container():
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                clean = prepare_survey_dataframe(raw)
                set_dataframe(clean, f"Uploaded: {uploaded.name}")
                st.success(f"Loaded {len(clean):,} rows from {uploaded.name}")
                df = clean
            except (DataValidationError, ValueError) as exc:
                st.error(f"Unable to load file: {exc}")

        if st.button("Use demo dataset", width="stretch"):
            demo_df = demo_loader()
            set_dataframe(demo_df, "Demo dataset")
            st.success("Demo data loaded")
            df = demo_df

    if df is None or df.empty:
        st.info("Load data to see profile details.")
        return

    profile = dataframe_profile(df)
    health = data_health(df)
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{profile['rows']:,}")
    cols[1].markdown(
        f"""
        <div class="kpi-card" title="Span of available responses in this dataset.">
            <div class="kpi-label">Date Range</div>
            <div class="kpi-value">{profile['date_range']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols[2].metric("Channels", profile["channels"])
    cols[3].metric("Regions", profile["regions"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        f"Missing NPS: {health.missing_nps_pct:.1f}% | Empty comments: {health.empty_comment_pct:.1f}%"
    )
    st.dataframe(df.head(10))


# ---------------------------------------------------------------------------
# Tab 2
# ---------------------------------------------------------------------------

def render_nps_tab(df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Display KPI cards and trend charts. Returns filtered data + KPI."""

    if df is None or df.empty:
        st.info("Upload or load data to view analytics.")
        return pd.DataFrame(), {}

    st.subheader("NPS Overview & Trends")
    filtered, _, channels, regions = _filter_controls(df, prefix="nps", title="Filters")
    if "nps_submit_token" not in st.session_state:
        st.session_state["nps_submit_token"] = 0
    submitted = st.button("Apply Filters", type="primary", key="nps_submit")
    if submitted:
        st.session_state["nps_submit_token"] += 1

    if st.session_state["nps_submit_token"] == 0:
        st.info("Configure filters and click Apply to view KPIs.")
        return pd.DataFrame(), {}

    cache_key = _cache_key("metrics", filtered, channels, regions) + f"::{st.session_state['nps_submit_token']}"
    metrics_bundle = _get_metrics_from_cache(cache_key, filtered)

    _render_kpis(metrics_bundle["kpi"])
    st.plotly_chart(nps_trend_chart(metrics_bundle["trend"]), width="stretch")

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        dimension_bar_chart(metrics_bundle["by_channel"], "channel"),
        width="stretch",
    )
    col2.plotly_chart(
        dimension_bar_chart(metrics_bundle["by_region"], "region"),
        width="stretch",
    )
    _render_keyword_cloud(top_keywords(filtered))
    return filtered, {"cache_key": cache_key, "metrics": metrics_bundle["kpi"]}


def _render_kpis(kpi: NPSKPI) -> None:
    cards = [
        ("Overall NPS", f"{kpi.overall_nps:.2f}", "Net Promoter Score: promoters minus detractors, scaled 0-100."),
        ("Promoters %", f"{kpi.promoter_pct:.2f}%", "Share of respondents scoring 9-10."),
        ("Passives %", f"{kpi.passive_pct:.2f}%", "Share of respondents scoring 7-8."),
        ("Detractors %", f"{kpi.detractor_pct:.2f}%", "Share of respondents scoring 0-6."),
        ("Responses", f"{kpi.total_responses:,}", "Total survey responses after filters."),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, tooltip) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="kpi-card" title="{tooltip}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 3
# ---------------------------------------------------------------------------

def render_themes_tab(
    df: pd.DataFrame,
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Display AI themes with filtering controls."""

    if df.empty:
        st.info("Select a dataset with comments to extract themes.")
        return []

    st.subheader("AI Themes & Sentiment")
    filtered, _, channels, regions = _filter_controls(df, prefix="themes", title="Filters")
    if "themes_submit_token" not in st.session_state:
        st.session_state["themes_submit_token"] = 0

    submitted = st.button("Show Themes", type="primary", key="themes_submit")
    if submitted:
        st.session_state["themes_submit_token"] += 1

    if st.session_state["themes_submit_token"] == 0:
        st.info("Adjust filters and click 'Show Themes' to extract insights.")
        return []
    if filtered.empty:
        st.warning("No records match the selected filters.")
        return []
    themes = _get_themes(filtered, channels, regions, llm, prefix="themes", submit_token=st.session_state["themes_submit_token"])

    if not themes:
        st.warning("No themes available for the selection.")
        return []

    cols = st.columns(2)
    for idx, theme in enumerate(themes):
        col = cols[idx % 2]
        badge_class = f"sentiment-{theme['sentiment'].lower()}"
        examples = "".join(f"<li>{comment}</li>" for comment in theme["example_comments"])
        col.markdown(
            f"""
            <div class="theme-card">
                <div class="theme-card-header">
                    <div>
                        <div class="theme-name">{theme['name']}</div>
                        <div class="theme-description">{theme['description']}</div>
                    </div>
                    <span class="theme-badge {badge_class}">{theme['sentiment']}</span>
                </div>
                <div class="theme-metrics">
                    <div><strong>{theme['volume']}</strong><span>Mentions ({theme['volume_pct']:.1f}%)</span></div>
                    <div><strong>{theme['avg_nps']:.2f}</strong><span>Avg NPS</span></div>
                </div>
                <div class="theme-comments">
                    <p>Sample comments:</p>
                    <ul>{examples}</ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return themes


# ---------------------------------------------------------------------------
# Tab 4
# ---------------------------------------------------------------------------

def render_comparison_tab(df: pd.DataFrame) -> dict[str, Any]:
    """Compare two date ranges."""

    if df.empty:
        st.info("Load data to compare against.")
        return {}

    st.subheader("What Changed?")
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    span_days = max((date_max - date_min).days, 1)
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown('<div class="filter-card-title">Comparison Filters</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        range_a = _select_time_range("comparison_a", date_min, date_max)
    with col_b:
        range_b = _select_time_range("comparison_b", date_min, date_max)
    col_channel, col_region = st.columns(2)
    with col_channel:
        channels = _multiselect_with_all(
            "Channels",
            sorted(df["channel"].unique()),
            widget_key="cmp_channels",
        )
    with col_region:
        regions = _multiselect_with_all(
            "Regions",
            sorted(df["region"].unique()),
            widget_key="cmp_regions",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if range_a == range_b:
        delta = range_b[1] - range_b[0]
        new_end = range_a[0] - timedelta(days=1)
        new_start = new_end - delta
        if new_start.date() < date_min:
            st.error("Select two distinct date ranges for Period A and Period B.")
            return {}
        range_b = (new_start, new_end)
        st.info("Period B auto-shifted to the previous window for comparison.")
    if "comparison_submit_token" not in st.session_state:
        st.session_state["comparison_submit_token"] = 0
    if st.button("Apply Comparison", type="primary", key="comparison_submit"):
        st.session_state["comparison_submit_token"] += 1

    if st.session_state["comparison_submit_token"] == 0:
        st.info("Adjust filters and click Apply Comparison to view results.")
        return {}

    subset_a = subset_for_filters(
        df,
        range_a,
        channels,
        regions,
    )
    subset_b = subset_for_filters(
        df,
        range_b,
        channels,
        regions,
    )

    metrics = comparison_nps(subset_a, subset_b)
    cols = st.columns(3)
    cols[0].metric("NPS A", f"{metrics['nps_a']:.2f}", delta=f"{metrics['responses_a']} responses")
    cols[1].metric("NPS B", f"{metrics['nps_b']:.2f}", delta=f"{metrics['responses_b']} responses")
    cols[2].metric("Δ NPS", f"{metrics['delta']:.2f}")

    breakdown = comparison_table(subset_a, subset_b, "channel")
    st.plotly_chart(comparison_bar_chart(breakdown, "channel"), width="stretch")
    st.dataframe(breakdown.round(2))
    return {
        "subset_a": subset_a,
        "subset_b": subset_b,
        "metrics": metrics,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Tab 5
# ---------------------------------------------------------------------------

def render_summary_tab(
    df: pd.DataFrame,
    llm: LLMClient,
    comparison: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate an executive summary. Returns the themes used."""

    if df.empty:
        st.info("Load data to summarize.")
        return []

    st.subheader("Executive Summary")
    date_range, period_label = _summary_time_range(df)
    if date_range is None:
        st.warning("Select a valid time window (max 60 days) to proceed.")
        return []

    channels = _multiselect_with_all(
        "Channels",
        sorted(df["channel"].unique()),
        widget_key="summary_channels",
    )
    regions = _multiselect_with_all(
        "Regions",
        sorted(df["region"].unique()),
        widget_key="summary_regions",
    )

    filtered = subset_for_filters(df, date_range, channels, regions)
    if filtered.empty:
        st.warning("No data matches the selected summary filters.")
        return []

    themes = _get_themes(filtered, channels, regions, llm, prefix="summary")
    limited_themes = themes[:5]
    comparison_context = _prepare_comparison_context(comparison)

    if st.button("Generate summary", width="stretch"):
        try:
            with st.spinner("Summarizing..."):
                summary = generate_executive_summary(
                    filtered, limited_themes, comparison_context, llm, period_label
                )
            st.markdown(summary)
        except Exception as exc:  # pragma: no cover - surfaces as UI error
            st.error(f"Unable to generate summary: {exc}")
    return limited_themes


# ---------------------------------------------------------------------------
# Tab 6
# ---------------------------------------------------------------------------

def render_chat_tab(
    df: pd.DataFrame,
    metrics: NPSKPI | None,
    themes: list[dict[str, Any]] | None,
    llm: LLMClient,
    chat_history: list[dict[str, str]],
    add_message: Callable[[str, str], None],
) -> None:
    """Chatbot interface."""

    st.subheader("Ask the Data")
    chat_container = st.container()
    for message in chat_history:
        if message["role"] == "user":
            chat_container.markdown(f"**You:** {message['content']}")
        else:
            chat_container.info(message["content"])

    chat_key = "chat_input"
    if st.session_state.get("chat_input_reset"):
        st.session_state[chat_key] = ""
        st.session_state["chat_input_reset"] = False
    question = st.text_input("Ask a question", key=chat_key)
    if st.button("Send", type="primary") and question:
        add_message("user", question)
        with st.spinner("Thinking..."):
            answer = answer_question(question, df, metrics, themes, llm)
        add_message("assistant", answer)
        st.session_state["chat_input_reset"] = True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _filter_controls(
    df: pd.DataFrame,
    prefix: str,
    title: str = "Filters",
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp] | None, list[str], list[str]]:
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="filter-card-title">{title}</div>', unsafe_allow_html=True)
    col_date, col_channel, col_region = st.columns([1.4, 1, 1])
    with col_date:
        date_range = _select_time_range(prefix, date_min, date_max)
        st.caption("Pick the window you want the KPIs to reflect.")
    with col_channel:
        channels = _multiselect_with_all(
            "Channels",
            sorted(df["channel"].unique()),
            widget_key=f"{prefix}_channels",
        )
    with col_region:
        regions = _multiselect_with_all(
            "Regions",
            sorted(df["region"].unique()),
            widget_key=f"{prefix}_regions",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = subset_for_filters(df, date_range, channels, regions)
    return filtered, date_range, list(channels), list(regions)


def _cache_key(prefix: str, df: pd.DataFrame, channels: Iterable[str], regions: Iterable[str]) -> str:
    fingerprint = hash(
        (
            len(df),
            df["date"].min().isoformat() if not df.empty else "0",
            df["date"].max().isoformat() if not df.empty else "0",
            tuple(channels),
            tuple(regions),
        )
    )
    return f"{prefix}::{fingerprint}"


def _get_metrics_from_cache(key: str, df: pd.DataFrame) -> dict[str, Any]:
    cache = st.session_state.get("metrics_cache", {})
    if key not in cache:
        cache[key] = {
            "kpi": compute_kpis(df),
            "trend": nps_trend(df),
            "by_channel": nps_by_dimension(df, "channel"),
            "by_region": nps_by_dimension(df, "region"),
        }
        st.session_state["metrics_cache"] = cache
    return cache[key]


def _get_themes(
    df: pd.DataFrame,
    channels: list[str],
    regions: list[str],
    llm: LLMClient,
    prefix: str,
    submit_token: int | None = None,
) -> list[dict[str, Any]]:
    cache_key = _cache_key(f"{prefix}_themes", df, channels, regions) + (f"::{submit_token}" if submit_token is not None else "")
    cache = st.session_state.get("theme_cache", {})
    if cache_key not in cache:
        with st.spinner("Computing themes..."):
            cache[cache_key] = extract_themes(df, llm)
        st.session_state["theme_cache"] = cache
    return cache[cache_key]


def _summary_time_range(df: pd.DataFrame) -> tuple[tuple[pd.Timestamp, pd.Timestamp] | None, str]:
    if df.empty:
        return None, "period"

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    options = list(TIME_PRESETS.keys())
    choice = st.selectbox(
        "Time Period",
        options,
        index=options.index("Last 7 days") if "Last 7 days" in options else 0,
        key="summary_period",
    )
    days = TIME_PRESETS[choice]
    if days:
        start = max(date_min, date_max - timedelta(days=days - 1))
        st.caption(f"Limiting summary context to the last {days} days.")
        return (pd.Timestamp(start), pd.Timestamp(date_max)), choice

    custom = st.date_input(
        "Custom range",
        value=(max(date_max - timedelta(days=6), date_min), date_max),
        min_value=date_min,
        max_value=date_max,
        key="summary_dates",
    )
    start, end = _ensure_tuple_range(custom)
    label = f"{(end - start).days + 1}-day period"
    return (pd.Timestamp(start), pd.Timestamp(end)), label


def _prepare_comparison_context(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if not comparison:
        return None
    context: dict[str, Any] = {}
    metrics = comparison.get("metrics")
    if metrics:
        context["metrics"] = metrics
    breakdown = comparison.get("breakdown")
    if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
        context["channel_shift"] = (
            breakdown[["channel", "delta_nps"]]
            .head(5)
            .to_dict(orient="records")
        )
    return context or None


def _multiselect_with_all(label: str, options: list[str], widget_key: str) -> list[str]:
    """Render a multiselect with checkboxes and an 'All' shortcut option."""

    all_label = f"All {label.lower()}"
    selection = st.multiselect(
        label,
        [all_label] + options,
        default=[all_label],
        key=widget_key,
    )
    if all_label in selection or not selection:
        return options
    return [value for value in selection if value != all_label]


def _select_time_range(prefix: str, date_min: date, date_max: date, default_choice: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    options = list(TIME_PRESETS.keys())
    if default_choice is None or default_choice not in options:
        default_choice = options[0]
    choice = st.selectbox(
        "Time Period",
        options,
        index=options.index(default_choice),
        key=f"{prefix}_period",
    )
    days = TIME_PRESETS[choice]
    if days:
        start = max(date_min, date_max - timedelta(days=days - 1))
        st.caption(f"Limiting data to the last {days} days.")
        return pd.Timestamp(start), pd.Timestamp(date_max)
    custom = st.date_input(
        "Custom range",
        value=(max(date_max - timedelta(days=6), date_min), date_max),
        min_value=date_min,
        max_value=date_max,
        key=f"{prefix}_dates",
    )
    start, end = _ensure_tuple_range(custom)
    return pd.Timestamp(start), pd.Timestamp(end)


def _ensure_tuple_range(selection) -> tuple[date, date]:
    if isinstance(selection, tuple) and len(selection) == 2:
        start, end = selection
    else:
        start = end = selection
    if isinstance(start, date) and isinstance(end, date) and start > end:
        start, end = end, start
    return start, end
def _render_keyword_cloud(keywords: list[dict[str, int]]) -> None:
    if not keywords:
        st.info("Keywords will appear once comments are available for this filter.")
        return

    max_count = max(k["count"] for k in keywords)
    palette = ["#d5ccff", "#b59dff", "#f7aef8", "#9ad0ff", "#c0ffee"]
    html_snippets: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(keywords):
        if item["word"] in seen:
            continue
        seen.add(item["word"])
        scale = 0.9 + (item["count"] / max_count) * 1.5
        color = palette[idx % len(palette)]
        html_snippets.append(
            f'<span style="font-size:{scale:.2f}rem;color:{color};">{item["word"]}</span>'
        )
    html = """
    <div class="keyword-cloud-card section-card">
        <div class="keyword-cloud-header">Top Keywords</div>
        <div class="keyword-cloud-words">
            {words}
        </div>
    </div>
    """.format(words=" ".join(html_snippets))
    st.markdown(html, unsafe_allow_html=True)
