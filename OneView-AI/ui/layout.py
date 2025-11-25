"""Streamlit layout helpers for the Multimodal Video → Analytics Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence
import pandas as pd
import streamlit as st

from components import cards
from core import visuals
from core.transcript import format_timestamp
from theme.style import sentiment_color


def _video_runtime(video_state: Dict, transcript_df: pd.DataFrame) -> float:
    duration_seconds = video_state.get("duration") or 0.0
    if (not duration_seconds or duration_seconds <= 0) and isinstance(transcript_df, pd.DataFrame) and not transcript_df.empty:
        duration_seconds = float(transcript_df["end"].max())
    return duration_seconds / 60.0


def render_overview_tab(
    video_state: Dict,
    transcript_df: pd.DataFrame,
) -> None:
    video_path = video_state.get("video_path")
    if video_path:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{video_state.get('video_name', Path(video_path).name)}**", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Add a file to begin the analysis.")


def render_timeline_tab(segment_df: pd.DataFrame, analytics_summary: Dict) -> None:
    if segment_df is None or segment_df.empty:
        st.info("Timeline analytics will appear after processing a video.")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Timeline Filters")
    min_time = float(segment_df["start"].min())
    max_time = float(segment_df["end"].max())
    time_range = st.slider(
        "Select time window (seconds)",
        min_value=0,
        max_value=int(max_time),
        value=(int(min_time), int(max_time)),
    )
    sentiment_cap = st.slider("Highlight segments above intensity", 0.0, 1.0, 0.3, 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = segment_df[(segment_df["start"] >= time_range[0]) & (segment_df["end"] <= time_range[1])]
    if filtered.empty:
        filtered = segment_df

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(visuals.sentiment_timeline(filtered), use_container_width=True)
    st.caption("Use the built-in range slider or adjust filters above to focus on specific minutes.")
    st.markdown("</div>", unsafe_allow_html=True)

    dual_cols = st.columns(2)
    with dual_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(visuals.energy_intensity(filtered), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with dual_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(visuals.keyword_heatmap(filtered), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_cols = st.columns(2)
    with bottom_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(visuals.sentiment_distribution(filtered), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with bottom_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.plotly_chart(visuals.segment_type_share(filtered), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if sentiment_cap:
        intense_segments = filtered[filtered["sentiment_score"].abs() >= sentiment_cap]
        if not intense_segments.empty:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Attention Heatmap (intense segments)")
            highlights = [
                f"- {format_timestamp(row.start)} — {row.segment_type.title()}: {row.snippet}"
                for row in intense_segments.itertuples()
            ]
            for highlight in highlights:
                st.write(highlight)
            st.markdown("</div>", unsafe_allow_html=True)


def render_key_moments_tab(key_moments: Sequence[Dict]) -> None:
    if not key_moments:
        st.info("Key moments require processed analytics and may leverage AI.")
        return
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Key Moments Timeline")
    for idx, moment in enumerate(key_moments):
        tags = moment.get("tags") or [moment.get("segment_type", "moment")]
        cards.timeline_card(
            timestamp=moment.get("timestamp", "--"),
            title=moment.get("title", f"Moment {idx+1}"),
            description=moment.get("description", ""),
            sentiment=moment.get("sentiment", "neutral"),
            tags=[tag for tag in tags if tag],
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_transcript_tab(transcript_df: pd.DataFrame) -> None:
    if transcript_df is None or transcript_df.empty:
        st.info("Transcript will appear here after processing.")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Transcript Explorer")
    with st.container():
        st.markdown("<div class='sticky-toolbar'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        sentiment = col1.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])
        keyword = col2.text_input("Search transcript", placeholder="Filter by keyword, owner, risk…")
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = transcript_df
    if sentiment != "All" and "sentiment_label" in transcript_df.columns:
        filtered = filtered[filtered["sentiment_label"] == sentiment]
    if keyword:
        filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]

    badge_map = {
        "positive": "🟢 Positive",
        "neutral": "⚪ Neutral",
        "negative": "🔴 Negative",
    }
    display_df = filtered.copy()
    display_df["Start"] = display_df["start"].map(format_timestamp)
    display_df["End"] = display_df["end"].map(format_timestamp)
    display_df["Sentiment"] = display_df["sentiment_label"].map(lambda x: badge_map.get(x, ""))
    display_df = display_df[["Start", "End", "Sentiment", "text"]].rename(columns={"text": "Content"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Click a row to cross-reference with the timeline. Timestamp linking is in preview.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_summary_tab(bundle: Dict) -> None:
    if not bundle or not (bundle.get("summary") or bundle.get("insights")):
        st.info("Insights and action items will appear after processing.")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Executive Summary")
    for bullet in bundle.get("summary", bundle.get("insights", [])):
        st.write(f"- {bullet}")
    st.markdown("</div>", unsafe_allow_html=True)

    action_items = bundle.get("actions") or bundle.get("action_items") or []
    risks = bundle.get("risks", [])

    dual_cols = st.columns(2)
    with dual_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if action_items:
            cards.action_list(
                "Action Items",
                [f"{item.get('item', 'Action')} — Owner: {item.get('owner', 'Unknown')} (Due: {item.get('due', 'TBD')})" for item in action_items],
                color="#2563eb",
            )
        else:
            st.write("No action items identified.")
        st.markdown("</div>", unsafe_allow_html=True)
    with dual_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if risks:
            cards.action_list("Risks & Flags", risks, color="#f97316")
        else:
            st.write("No risks detected.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_chat_tab(
    chat_history: List[Dict[str, str]],
    *,
    on_submit,
    disabled: bool = False,
) -> None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Chat with the Video")
    quick_questions = [
        "Summarise this meeting",
        "List the riskiest moments",
        "Who owns the action items?",
    ]
    quick_cols = st.columns(len(quick_questions))
    for col, question in zip(quick_cols, quick_questions):
        if col.button(question, disabled=disabled, key=f"quick_{question}", use_container_width=True):
            on_submit(question)

    if not chat_history:
        st.info("Ask targeted questions once analytics are ready. Responses cite relevant timestamps.")

    for message in chat_history:
        role = message.get("role", "assistant")
        display_role = "assistant" if role == "assistant" else "user"
        with st.chat_message(display_role):
            timestamp = message.get("timestamp")
            if timestamp:
                st.caption(timestamp)
            st.markdown(message.get("content", "") or "_No content returned._")

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask the assistant",
            disabled=disabled,
            placeholder="e.g., Highlight decisions in last 3 minutes",
        )
        submitted = st.form_submit_button("Send", use_container_width=True, disabled=disabled)
        if submitted and question:
            with st.spinner("Video Assistant drafting a response..."):
                on_submit(question.strip())
    st.markdown("</div>", unsafe_allow_html=True)


def render_processing_banner(state: str) -> None:
    """Show animated processing banner."""
    if not state:
        return
    st.markdown(
        f"<div class='processing-banner'><div class='processing-dot'></div><div><strong>Pipeline</strong><br>{state}</div></div>",
        unsafe_allow_html=True,
    )


def render_flow_progress(flow: Dict, talk: Dict) -> None:
    """Display a compact flow with step statuses."""
    steps = [
        ("upload", "File Upload"),
        ("data", "Data Extraction"),
        ("video", "Video Analysis"),
        ("insights", "Insight Synthesis"),
        ("avatar", "Avatar Render"),
        ("delivery", "Delivery & Share"),
    ]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Flow Progress")
    cols = st.columns(len(steps))
    for col, (key, label) in zip(cols, steps):
        status = flow.get(key, "pending")
        if status == "pending":
            width = "12%"
            color = "rgba(255,255,255,0.16)"
        elif status == "in_progress":
            width = "55%"
            color = "#3b82f6"
        else:
            width = "100%"
            color = "#22c55e"
        col.markdown(
            f"<div style='text-align:center;font-size:0.9rem;color:#e8f1ff;'>"
            f"<div style='margin-bottom:6px;'>{label}</div>"
            f"<div style='height:6px;border-radius:6px;background:rgba(255,255,255,0.08);'>"
            f"<div style='height:6px;border-radius:6px;width:{width};"
            f"background:{color};transition:width 0.4s ease, background 0.4s ease;'></div></div>"
            f"<div style='margin-top:6px;font-size:0.8rem;color:#9fb3d2;'>{status.replace('_',' ').title()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if talk.get("status") and talk.get("status") != "done":
        st.info(f"Avatar render status: {talk.get('status')}")


def render_delivery_panel(
    bundle: Dict,
    talk: Dict,
    *,
    on_email=None,
) -> None:
    """Display final insights, actions, and D-ID delivery."""
    if not bundle:
        st.info("Run the multimodal pipeline to populate insights and avatar delivery.")
        return

    st.markdown("<div class='card gradient-card'>", unsafe_allow_html=True)
    st.markdown("#### Multimodal Insights")
    for insight in bundle.get("insights", []):
        st.write(f"- {insight}")
    st.markdown("</div>", unsafe_allow_html=True)

    actions = bundle.get("actions") or []
    risks = bundle.get("risks") or []
    dual_cols = st.columns(2)
    with dual_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if actions:
            cards.action_list(
                "Action Items",
                [f"{item.get('item', 'Action')} — Owner: {item.get('owner', 'Unknown')} (Due: {item.get('due', 'TBD')})" for item in actions],
                color="#2563eb",
            )
        else:
            st.write("No actions detected.")
        st.markdown("</div>", unsafe_allow_html=True)
    with dual_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if risks:
            cards.action_list("Risks & Flags", risks, color="#f97316")
        else:
            st.write("No risks detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Avatar Delivery (D-ID)")
    status = talk.get("status")
    url = talk.get("result_url")
    if not status:
        st.caption("Trigger the pipeline to request a D-ID talk. Status updates will show here.")
    if status:
        st.info(f"D-ID status: {status}")
    if url:
        with st.container():
            st.markdown("<div style='max-width:560px;'>", unsafe_allow_html=True)
            st.video(url)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"[Download video]({url})", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.button("Download Package", disabled=not url, use_container_width=True)
    with cols[1]:
        if st.button("Send via Email", disabled=on_email is None, use_container_width=True):
            if callable(on_email):
                on_email()
            st.success("Email sent (stub).")
    st.markdown("</div>", unsafe_allow_html=True)


def render_upload_preview(assets: List[Dict]) -> None:
    """Display uploaded files and video preview."""
    if not assets:
        return
    # Deduplicate by path to avoid double rendering
    unique_assets = {item["path"]: item for item in assets}.values()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Uploaded Assets")
    video_items = [a for a in unique_assets if a.get("kind") == "video"]
    other_items = [a for a in unique_assets if a.get("kind") != "video"]
    cols = st.columns(2)
    with cols[0]:
        if video_items:
            st.caption("Video")
            st.video(video_items[-1]["path"])
    with cols[1]:
        if other_items:
            st.caption("Documents & Media")
            chips = " ".join(f"<span class='chip'>{item['name']} · {item['kind']}</span>" for item in other_items)
            st.markdown(f"<div class='upload-chips'>{chips}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
