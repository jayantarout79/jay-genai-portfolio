"""Data-aware chatbot orchestration."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from typing import Any

import pandas as pd
from langchain.prompts import ChatPromptTemplate

from ai.llm_client import LLMClient
from core.data_loader import filter_dataframe
from core.metrics import NPSKPI, compute_kpis, nps_by_dimension


def answer_question(
    question: str,
    df: pd.DataFrame,
    precomputed_metrics: NPSKPI | None,
    precomputed_themes: list[dict[str, Any]] | None,
    llm: LLMClient,
) -> str:
    """Return a grounded response to the user's analytical question."""

    if df.empty:
        return "No survey data is available yet. Upload a CSV to begin."

    filters = _infer_filters(question, df)
    filtered_df = filter_dataframe(
        df,
        date_range=filters.get("date_range"),
        channels=filters.get("channels"),
        regions=filters.get("regions"),
    )
    metrics = precomputed_metrics or compute_kpis(filtered_df)
    top_channels = nps_by_dimension(filtered_df, "channel").to_dict(orient="records")
    top_regions = nps_by_dimension(filtered_df, "region").to_dict(orient="records")
    context = {
        "filters": filters,
        "kpis": asdict(metrics),
        "channel_breakdown": top_channels[:5],
        "region_breakdown": top_regions[:5],
        "themes": precomputed_themes or [],
    }

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a precise CX analytics assistant."),
            (
                "user",
                "Question: {question}\nStructured data: {context}\n"
                "Provide a concise answer (<= 6 sentences) referencing numbers.",
            ),
        ]
    )
    prompt = prompt_template.format(question=question, context=json.dumps(context))

    if llm and llm.is_configured():
        return llm.generate_text(prompt, temperature=0.3, max_tokens=500)

    return _offline_answer(question, context)


def _infer_filters(question: str, df: pd.DataFrame) -> dict[str, Any]:
    question_lower = question.lower()
    channels = [c for c in df["channel"].unique() if c.lower() in question_lower]
    regions = [r for r in df["region"].unique() if r.lower() in question_lower]
    date_range = None
    period_match = re.search(r"last (\w+)", question_lower)
    if period_match:
        # Placeholder: let the UI filters drive precise dates; we only tag intent here.
        date_range = None
    return {"channels": channels or None, "regions": regions or None, "date_range": date_range}


def _offline_answer(question: str, context: dict[str, Any]) -> str:
    kpis = context.get("kpis", {})
    channels = context.get("channel_breakdown", [])
    regions = context.get("region_breakdown", [])
    parts = [
        f"Overall NPS is {kpis.get('overall_nps', 0)} with {kpis.get('total_responses', 0)} responses.",
    ]
    if channels:
        top = channels[0]
        parts.append(
            f"Top channel: {top.get('channel', 'N/A')} at {top.get('nps', 0):.1f} NPS."
        )
    if regions:
        top_region = regions[0]
        parts.append(
            f"Regional leader: {top_region.get('region', 'N/A')} at {top_region.get('nps', 0):.1f}."
        )
    themes = context.get("themes", [])
    if themes:
        parts.append(
            "Representative theme: "
            + themes[0].get("name", "Unnamed")
            + " — "
            + themes[0].get("description", "")
        )
    parts.append(
        "Hint: add an API key to unlock the conversational AI answer."
    )
    return " ".join(parts)
