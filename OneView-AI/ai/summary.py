"""Summary, action item, and risk generation helpers."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from core.transcript import format_timestamp


def _format_context(segment_df: pd.DataFrame, limit: int = 30) -> str:
    snippets: List[str] = []
    for row in segment_df.itertuples():
        snippets.append(
            f"[{format_timestamp(row.start)}-{format_timestamp(row.end)}] "
            f"{row.sentiment_label.upper()} • {', '.join(row.topics) or 'General'} • {row.snippet}"
        )
        if len(snippets) >= limit:
            break
    return "\n".join(snippets)


def generate_summary_bundle(segment_df: pd.DataFrame, llm_client, max_words: int = 220) -> Dict:
    """Return structured summary, action items, and risks."""
    if segment_df is None or segment_df.empty:
        return {"summary": [], "action_items": [], "risks": []}

    if llm_client is None or not getattr(llm_client, "available", False):
        placeholder = "Summary unavailable without OpenAI credentials."
        return {
            "summary": [placeholder],
            "action_items": [],
            "risks": [],
        }

    system_prompt = (
        "You are an operations analyst who turns meeting transcripts into concise executive readouts. "
        "Summaries must stay under 220 words, bullet-style, decision-focused."
    )
    context_block = _format_context(segment_df)
    user_prompt = (
        "Transcript analytics context:\n"
        f"{context_block}\n\n"
        "Provide JSON with fields summary (array of 5-7 bullet strings <= 220 words total), "
        "action_items (array of objects with item, owner, due fields—owner can be Unknown), "
        "and risks (array of short bullet strings)."
    )
    response = llm_client.response_json(system_prompt, user_prompt, max_output_tokens=800)
    summary_points = response.get("summary") or []
    if isinstance(summary_points, str):
        summary_points = [summary_points]
    action_items = response.get("action_items") or []
    risks = response.get("risks") or []
    if isinstance(risks, str):
        risks = [risks]
    return {
        "summary": summary_points,
        "action_items": action_items,
        "risks": risks,
    }

