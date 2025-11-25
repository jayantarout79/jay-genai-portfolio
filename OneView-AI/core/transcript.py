"""Transcription helpers that turn audio into structured transcript segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence
import logging

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Small chunk of transcript text with precise timing."""

    start: float
    end: float
    text: str


def _value_from_segment(segment: Any, key: str) -> float | str:
    """Safely retrieve values from dict-like or object segments."""
    if isinstance(segment, dict):
        return segment.get(key)
    return getattr(segment, key, None)


def transcribe_audio(audio_path: Path, llm_client) -> List[TranscriptSegment]:
    """Invoke the LLM client to obtain Whisper-style transcript segments."""
    if llm_client is None or not getattr(llm_client, "available", False):
        raise RuntimeError("OpenAI client unavailable. Set OPENAI_API_KEY to enable transcription.")
    raw_segments = llm_client.transcribe(audio_path)
    segments: List[TranscriptSegment] = []
    for segment in raw_segments:
        if isinstance(segment, TranscriptSegment):
            segments.append(segment)
            continue
        start = _value_from_segment(segment, "start")
        end = _value_from_segment(segment, "end")
        text = _value_from_segment(segment, "text")
        segments.append(
            TranscriptSegment(
                start=float(start or 0.0),
                end=float(end or 0.0),
                text=str(text or "").strip(),
            )
        )
    logger.info("Received %s transcript segments", len(segments))
    return segments


def segments_to_dataframe(segments: Sequence[TranscriptSegment]) -> pd.DataFrame:
    """Convert transcript segments into a DataFrame for downstream analytics."""
    data = [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "duration": max(seg.end - seg.start, 0.0),
        }
        for seg in segments
        if seg.text
    ]
    return pd.DataFrame(data)


def format_timestamp(seconds: float) -> str:
    """Format seconds as mm:ss for display."""
    seconds = max(seconds, 0.0)
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
