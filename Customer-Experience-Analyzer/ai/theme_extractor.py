"""Theme extraction utilities powered by the LLM."""
from __future__ import annotations

import json
import logging
from textwrap import dedent
from typing import Any

import pandas as pd

from ai.llm_client import LLMClient


def extract_themes(
    df: pd.DataFrame,
    llm: LLMClient,
    max_themes: int = 8,
) -> list[dict[str, Any]]:
    """Use an LLM (with heuristics fallback) to extract comment themes."""

    fields = ["comment_text", "nps_score"]
    if "product_category" in df.columns:
        fields.append("product_category")
    if "segment" in df.columns:
        fields.append("segment")
    comments = df[fields].dropna(subset=["comment_text", "nps_score"])
    comments = comments[comments["comment_text"].str.len() > 0]
    if comments.empty:
        return []

    prompt = dedent(
        f"""
        You are helping a CX analytics team understand survey feedback.
        Analyze the following customer comments and respond ONLY with a JSON array
        (no prose, no markdown). The array should contain up to {max_themes} themes.
        Each theme object must include: name, description, volume, volume_pct,
        avg_nps, sentiment (Positive/Neutral/Negative) and two example_comments.

        Comments (JSON array of objects with score and text):
        {comments.head(250).to_json(orient='records')}
        """
    )

    payload = None
    if llm and llm.is_configured():
        response = llm.generate_text(prompt)
        payload = _parse_theme_payload(response)

    if not payload:
        logging.info("Falling back to heuristic theme extraction")
        return _heuristic_themes(comments, max_themes)

    return _enrich_theme_payload(payload, comments)


def _parse_theme_payload(text: str) -> list[dict[str, Any]] | None:
    try:
        snippet = text.strip()
        if "```" in snippet:
            parts = snippet.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") and part.endswith("]"):
                    return json.loads(part)
        start = snippet.find("[")
        end = snippet.rfind("]")
        if start == -1 or end == -1:
            return None
        return json.loads(snippet[start : end + 1])
    except json.JSONDecodeError:
        logging.info("Unable to parse theme JSON from LLM response; using fallback.")
        return None


def _enrich_theme_payload(payload: list[dict[str, Any]], comments: pd.DataFrame) -> list[dict[str, Any]]:
    total_comments = len(comments)
    cleaned: list[dict[str, Any]] = []
    for item in payload:
        volume = int(item.get("volume", 0))
        volume_pct = (volume / total_comments) * 100 if total_comments else 0
        avg_nps = float(item.get("avg_nps", 0))
        sentiment = (item.get("sentiment") or "").title()
        if sentiment not in {"Positive", "Neutral", "Negative"}:
            sentiment = _sentiment_from_score(avg_nps)
        elif sentiment == "Neutral":
            sentiment = _sentiment_from_score(avg_nps)
        cleaned.append(
            {
                "name": item.get("name", "Unnamed Theme"),
                "description": item.get("description", ""),
                "volume": volume,
                "volume_pct": round(volume_pct, 1),
                "avg_nps": round(avg_nps, 2),
                "sentiment": sentiment,
                "example_comments": item.get("example_comments", [])[:2],
            }
        )
    return cleaned


def _heuristic_themes(df: pd.DataFrame, max_themes: int) -> list[dict[str, Any]]:
    total = len(df)
    if total == 0:
        return []

    if "product_category" in df.columns:
        dimension = "product_category"
        label_template = "Product: {name}"
        description_template = "Feedback referencing {name}."
    elif "segment" in df.columns:
        dimension = "segment"
        label_template = "Segment: {name}"
        description_template = "Comments from the {name} segment."
    else:
        dimension = None

    summaries = []
    if dimension:
        grouped = df.groupby(dimension)
        for name, group in grouped:
            avg = group["nps_score"].mean()
            sentiment = _sentiment_from_score(avg)
            summaries.append(
                {
                    "name": label_template.format(name=name),
                    "description": description_template.format(name=name),
                    "volume": len(group),
                    "volume_pct": round((len(group) / total) * 100, 1),
                    "avg_nps": round(avg, 1),
                    "sentiment": sentiment,
                    "example_comments": group["comment_text"].head(2).tolist(),
                }
            )
        summaries.sort(key=lambda item: item["volume"], reverse=True)
        return summaries[:max_themes]

    avg = df["nps_score"].mean()
    return [
        {
            "name": "Overall feedback",
            "description": "Theme generated from all available comments.",
            "volume": total,
            "volume_pct": 100.0,
            "avg_nps": round(avg, 1),
            "sentiment": _sentiment_from_score(avg),
            "example_comments": df["comment_text"].head(2).tolist(),
        }
    ]


def _sentiment_from_score(score: float) -> str:
    if score >= 7.5:
        return "Positive"
    if score <= 5.5:
        return "Negative"
    return "Neutral"
