"""Thin wrapper around the OpenAI SDK for transcription and reasoning calls."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency in CI
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper that centralizes OpenAI interactions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        transcription_model: str = "whisper-1",
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.available = bool(self.api_key and OpenAI is not None)
        self.model = model
        self.transcription_model = transcription_model
        self.temperature = temperature
        self._client = OpenAI(api_key=self.api_key) if self.available else None

    def _ensure_client(self):
        if not self.available or self._client is None:
            raise RuntimeError("OpenAI client unavailable. Set OPENAI_API_KEY to enable AI features.")
        return self._client

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        client = self._ensure_client()
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError as exc:
            if json_mode:
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return (content or "").strip()

    def transcribe(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Send audio to Whisper and return verbose segments."""
        client = self._ensure_client()
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=self.transcription_model,
                response_format="verbose_json",
                file=audio_file,
            )
        segments = getattr(response, "segments", None)
        if segments is None:
            return []

        normalized = []
        for seg in segments:
            if hasattr(seg, "start"):
                normalized.append(
                    {
                        "start": float(getattr(seg, "start", 0.0)),
                        "end": float(getattr(seg, "end", 0.0)),
                        "text": getattr(seg, "text", ""),
                    }
                )
            elif isinstance(seg, dict):
                normalized.append(
                    {
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": str(seg.get("text", "")),
                    }
                )
        return normalized

    def response_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: int = 600,
    ) -> str:
        """Create a text response."""
        temperature = temperature if temperature is not None else self.temperature
        return self._chat_completion(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_output_tokens,
            json_mode=False,
        )

    def response_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: int = 800,
    ) -> Dict[str, Any]:
        """Create a JSON-structured response."""
        temperature = temperature if temperature is not None else self.temperature
        output_text = self._chat_completion(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_output_tokens,
            json_mode=True,
        )
        if not output_text:
            return {}
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - best effort
            logger.warning("Failed to decode JSON response: %s", exc)
            return {}

    def segment_profile(self, text: str, start: float, end: float) -> Dict[str, Any]:
        """Obtain a structured sentiment/topic analysis for a transcript segment."""
        prompt = (
            "You are a communications analyst. Review the segment and respond with JSON "
            "fields sentiment_score (-1 to 1 float), sentiment_label (positive|neutral|negative), "
            "topics (list of 1-4 concise nouns), segment_type (update|issue|decision|question|other), "
            "and insight (<=30 words)."
        )
        user = (
            f"Segment start: {start:.1f}s\nSegment end: {end:.1f}s\n"
            f"Transcript:\n{text.strip()}\n"
        )
        analysis = self.response_json(prompt, user, max_output_tokens=400)
        if "sentiment_score" in analysis:
            try:
                analysis["sentiment_score"] = float(analysis["sentiment_score"])
            except (TypeError, ValueError):
                analysis["sentiment_score"] = 0.0
        return analysis
