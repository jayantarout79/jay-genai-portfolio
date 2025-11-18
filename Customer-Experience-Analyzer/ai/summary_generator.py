"""Executive summary generation."""
from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd

from ai.llm_client import LLMClient
from core.metrics import compute_overall_nps, nps_by_dimension


def _summary_context(df: pd.DataFrame, themes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall_nps": round(compute_overall_nps(df), 1) if not df.empty else 0,
        "total_responses": len(df),
        "top_channels": nps_by_dimension(df, "channel").head(3).to_dict(
            orient="records"
        )
        if not df.empty
        else [],
        "themes": themes,
    }


def generate_executive_summary(
    df: pd.DataFrame,
    themes: list[dict[str, Any]],
    comparison_info: dict[str, Any] | None,
    llm: LLMClient,
    period_label: str = "period",
) -> str:
    """Use the LLM to craft an executive-ready summary."""

    context = _summary_context(df, themes)
    context["comparison"] = comparison_info or {}
    context["period_label"] = period_label
    prompt = dedent(
        f"""
        You are a Chief Customer Officer preparing an executive update focused on the selected {period_label}.
        Write a concise (<= 200 words) summary using the structured data below.
        Mention overall NPS and total responses, highlight 1-2 positive and
        negative themes, and call out material changes week-over-week if
        comparison data exists. Conclude with 2 recommended focus actions.

        DATA:
        {json.dumps(_json_ready(context))}
        """
    )

    if llm and llm.is_configured():
        return llm.generate_text(prompt, temperature=0.1, max_tokens=400)

    return _offline_summary(context)


def _offline_summary(context: dict[str, Any]) -> str:
    overall = context.get("overall_nps", 0)
    responses = context.get("total_responses", 0)
    themes = context.get("themes", [])
    pos = [t for t in themes if t.get("sentiment") == "Positive"]
    neg = [t for t in themes if t.get("sentiment") == "Negative"]
    lines = [
        f"Overall NPS currently sits at {overall:.1f} based on {responses} responses.",
    ]
    if pos:
        lines.append(
            "Positive momentum: "
            + ", ".join(t.get("name", "") for t in pos[:2])
        )
    if neg:
        lines.append(
            "Key friction points: "
            + ", ".join(t.get("name", "") for t in neg[:2])
        )
    lines.append("Recommended focus: double down on promoters and close the loop with detractors.")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    """Convert pandas / numpy objects into JSON serializable types."""

    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.DataFrame):
        records = value.to_dict(orient="records")
        return [_json_ready(record) for record in records]
    if isinstance(value, pd.Series):
        return {k: _json_ready(v) for k, v in value.to_dict().items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):  # type: ignore[attr-defined]
        return [_json_ready(v) for v in value.tolist()]
    return value
