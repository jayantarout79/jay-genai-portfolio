"""Conversational Q&A helper grounded in transcript analytics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from core.transcript import format_timestamp


def _format_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def answer_question(
    question: str,
    *,
    segment_df: pd.DataFrame,
    summary_bundle: Dict,
    key_moments: List[Dict],
    analytics_summary: Dict,
    llm_client,
) -> str:
    """Answer a question grounded in transcript segments and analytics."""
    if llm_client is None or not getattr(llm_client, "available", False):
        return "Chat assistant requires an OpenAI API key."

    recent_segments = []
    if segment_df is not None and not segment_df.empty:
        tail_df = segment_df.tail(12)
        for row in tail_df.itertuples():
            recent_segments.append(
                f"{format_timestamp(row.start)} ({row.sentiment_label}) [{row.segment_type}] {row.snippet}"
            )

    summary_text = _format_list(summary_bundle.get("summary", []))
    action_text = "\n".join(
        f"- {item.get('item', 'N/A')} (Owner: {item.get('owner', 'Unknown')})"
        for item in summary_bundle.get("action_items", [])
    )
    risk_text = _format_list(summary_bundle.get("risks", []))
    key_moment_text = "\n".join(
        f"- {moment.get('timestamp')} {moment.get('title')}: {moment.get('description')}"
        for moment in key_moments
    )
    analytics_text = _format_list(analytics_summary.get("insights", []))

    user_prompt = (
        f"Summary:\n{summary_text or 'N/A'}\n\n"
        f"Action Items:\n{action_text or 'N/A'}\n\n"
        f"Risks:\n{risk_text or 'N/A'}\n\n"
        f"Key Moments:\n{key_moment_text or 'N/A'}\n\n"
        f"Recent Transcript Segments:\n{chr(10).join(recent_segments) or 'N/A'}\n\n"
        f"Analytics Insights:\n{analytics_text or 'N/A'}\n\n"
        f"User question: {question}"
    )
    system_prompt = (
        "You are the Video Assistant, answering questions strictly with the provided context. "
        "Cite timestamps when relevant, note uncertainty when the context lacks details, "
        "and keep responses under 200 words."
    )
    answer = llm_client.response_text(system_prompt, user_prompt, max_output_tokens=500)
    return answer or "I'm sorry, I could not generate an answer."

