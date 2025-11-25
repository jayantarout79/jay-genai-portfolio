"""Key moment extraction using analytics context."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.transcript import format_timestamp


def _fallback_key_moments(segment_df: pd.DataFrame) -> List[Dict]:
    if segment_df is None or segment_df.empty:
        return []
    top_segments = segment_df.reindex(
        segment_df["sentiment_score"].abs().sort_values(ascending=False).index
    ).head(4)
    key_moments = []
    for row in top_segments.itertuples():
        key_moments.append(
            {
                "timestamp": format_timestamp(row.start),
                "title": f"{row.segment_type.title()} Highlight",
                "description": row.snippet,
                "sentiment": row.sentiment_label,
            }
        )
    return key_moments


def generate_key_moments(segment_df: pd.DataFrame, analytics_summary: Dict, llm_client) -> List[Dict]:
    """Use the LLM to extract notable timestamps and descriptions."""
    if segment_df is None or segment_df.empty:
        return []

    if llm_client is None or not getattr(llm_client, "available", False):
        return _fallback_key_moments(segment_df)

    system_prompt = (
        "You are a narrative analyst surfacing pivotal meeting moments. "
        "Return JSON with key_moments array of objects containing timestamp (MM:SS), "
        "title (<10 words), description (1-2 sentences), and sentiment (positive|neutral|negative)."
    )
    context_lines = []
    for row in segment_df.itertuples():
        context_lines.append(
            f"{format_timestamp(row.start)} {row.segment_type.upper()} {row.sentiment_label}: {row.snippet}"
        )
        if len(context_lines) >= 35:
            break
    insights = "\n".join(analytics_summary.get("insights", []))
    user_prompt = (
        f"Context segments:\n{chr(10).join(context_lines)}\n\n"
        f"High-level analytics:\n{insights or 'N/A'}"
    )
    response = llm_client.response_json(system_prompt, user_prompt, max_output_tokens=900)
    key_moments = response.get("key_moments")
    if isinstance(key_moments, dict):
        key_moments = [key_moments]
    if not key_moments:
        return _fallback_key_moments(segment_df)
    return key_moments

