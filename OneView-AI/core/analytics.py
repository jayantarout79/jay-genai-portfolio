"""Transcript segmentation and analytics computation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the",
    "and",
    "that",
    "have",
    "with",
    "this",
    "from",
    "they",
    "would",
    "there",
    "their",
    "about",
    "could",
    "should",
    "will",
    "your",
    "into",
    "while",
    "were",
    "being",
    "after",
    "before",
    "because",
    "what",
    "when",
    "where",
    "which",
    "whom",
    "whose",
    "here",
    "have",
    "also",
    "only",
    "have",
    "much",
    "many",
    "very",
    "just",
    "like",
    "need",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "upbeat",
    "win",
    "success",
    "improve",
    "best",
    "progress",
    "confident",
    "excited",
    "glad",
    "happy",
    "safe",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "negative",
    "issue",
    "problem",
    "delay",
    "risk",
    "concern",
    "angry",
    "sad",
    "worried",
    "stuck",
    "blocked",
    "fail",
    "broken",
}


def segment_transcript(transcript_df: pd.DataFrame, chunk_seconds: int = 45) -> pd.DataFrame:
    """Segment the transcript into near-equal duration blocks."""
    if transcript_df is None or transcript_df.empty:
        return pd.DataFrame(
            columns=["segment_id", "start", "end", "text", "duration", "snippet"]
        )

    segments: List[Dict] = []
    buffer: List[str] = []
    chunk_start = float(transcript_df.iloc[0]["start"])
    current_end = chunk_start
    segment_id = 1

    for row in transcript_df.itertuples():
        start = float(row.start)
        end = float(row.end)
        text = str(row.text).strip()
        if not text:
            continue

        if start - chunk_start >= chunk_seconds and buffer:
            full_text = " ".join(buffer).strip()
            segments.append(
                {
                    "segment_id": segment_id,
                    "start": chunk_start,
                    "end": current_end,
                    "text": full_text,
                    "duration": max(current_end - chunk_start, 0.0),
                    "snippet": f"{full_text[:160]}{'…' if len(full_text) > 160 else ''}",
                }
            )
            buffer = []
            chunk_start = start
            segment_id += 1

        buffer.append(text)
        current_end = end

    if buffer:
        full_text = " ".join(buffer).strip()
        segments.append(
            {
                "segment_id": segment_id,
                "start": chunk_start,
                "end": current_end,
                "text": full_text,
                "duration": max(current_end - chunk_start, 0.0),
                "snippet": f"{full_text[:160]}{'…' if len(full_text) > 160 else ''}",
            }
        )

    return pd.DataFrame(segments)


def _heuristic_sentiment(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    positives = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negatives = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    total = positives + negatives
    if total == 0:
        return 0.0
    score = (positives - negatives) / total
    return float(np.clip(score, -1.0, 1.0))


def _sentiment_label(score: float) -> str:
    if score > 0.25:
        return "positive"
    if score < -0.25:
        return "negative"
    return "neutral"


def _collect_topics(text: str, limit: int = 4) -> List[str]:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 3]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(limit)]


def _classify_segment(text: str) -> str:
    lowered = text.lower()
    if "?" in text or lowered.startswith(("how", "what", "when", "why")):
        return "question"
    if any(word in lowered for word in ("decision", "approved", "agreed")):
        return "decision"
    if any(word in lowered for word in ("issue", "problem", "blocked", "risk")):
        return "issue"
    if any(word in lowered for word in ("next", "todo", "action", "owner")):
        return "update"
    return "other"


def enrich_segments(segment_df: pd.DataFrame, llm_client) -> pd.DataFrame:
    """Add sentiment, topics, and lightweight classifications to each segment."""
    if segment_df is None or segment_df.empty:
        return segment_df

    enriched_rows = []
    for row in segment_df.itertuples():
        base = {
            "segment_id": row.segment_id,
            "start": float(row.start),
            "end": float(row.end),
            "duration": float(row.duration),
            "text": row.text,
            "snippet": row.snippet,
        }
        analysis = {}
        if llm_client is not None and getattr(llm_client, "available", False):
            try:
                analysis = llm_client.segment_profile(row.text, row.start, row.end)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("LLM segment analysis failed: %s", exc)
                analysis = {}
        if not analysis:
            score = _heuristic_sentiment(row.text)
            analysis = {
                "sentiment_score": score,
                "sentiment_label": _sentiment_label(score),
                "topics": _collect_topics(row.text),
                "segment_type": _classify_segment(row.text),
                "insight": row.snippet,
            }
        analysis.setdefault("sentiment_label", _sentiment_label(analysis.get("sentiment_score", 0.0)))
        analysis.setdefault("topics", _collect_topics(row.text))
        analysis.setdefault("segment_type", _classify_segment(row.text))
        analysis.setdefault("insight", row.snippet)
        enriched_rows.append({**base, **analysis})
    return pd.DataFrame(enriched_rows)


def compute_aggregates(segment_df: pd.DataFrame) -> Dict:
    """Compute high-level analytics from enriched segments."""
    if segment_df is None or segment_df.empty:
        return {
            "avg_sentiment": 0.0,
            "sentiment_distribution": {},
            "topic_frequency": {},
            "key_moment_threshold": 0.45,
            "key_segment_ids": [],
            "insights": [],
        }

    avg_sentiment = float(segment_df["sentiment_score"].mean())
    distribution = segment_df["sentiment_label"].value_counts().to_dict()

    topic_counts: Counter = Counter()
    for topics in segment_df["topics"]:
        if isinstance(topics, Sequence):
            topic_counts.update([topic for topic in topics if topic])

    threshold = 0.45
    key_segments = segment_df.loc[segment_df["sentiment_score"].abs() >= threshold]
    insights = []
    if not key_segments.empty:
        dip = key_segments.sort_values("sentiment_score").iloc[0]
        peak = key_segments.sort_values("sentiment_score", ascending=False).iloc[0]
        insights.append(
            f"Sentiment dipped near minute {dip['start']/60:.1f} ({dip['sentiment_label']})."
        )
        insights.append(
            f"Highest energy around minute {peak['start']/60:.1f} discussing {', '.join(peak['topics']) or 'key topics'}."
        )

    return {
        "avg_sentiment": avg_sentiment,
        "sentiment_distribution": distribution,
        "topic_frequency": dict(topic_counts.most_common(10)),
        "key_moment_threshold": threshold,
        "key_segment_ids": key_segments["segment_id"].tolist(),
        "insights": insights,
    }


def annotate_transcript(transcript_df: pd.DataFrame) -> pd.DataFrame:
    """Add heuristic sentiment labels to the raw transcript for filtering."""
    if transcript_df is None or transcript_df.empty:
        return transcript_df
    enriched = transcript_df.copy()
    enriched["sentiment_score"] = enriched["text"].apply(_heuristic_sentiment)
    enriched["sentiment_label"] = enriched["sentiment_score"].apply(_sentiment_label)
    return enriched
