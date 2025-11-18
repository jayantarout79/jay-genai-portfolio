"""Thin wrapper around the OpenAI API with graceful degradation."""
from __future__ import annotations

import json
import logging
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


class LLMClient:
    """Encapsulates LLM access with graceful fallbacks."""

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None
        if api_key and OpenAI:
            self._client = OpenAI(api_key=api_key)
        elif api_key and not OpenAI:
            logging.warning("openai package not installed; LLM disabled")

    def is_configured(self) -> bool:
        return self._client is not None

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        """Send a prompt to the configured model and return the text."""

        if not self._client:
            logging.warning("LLM client not configured; falling back to stub response")
            return self._offline_stub(prompt)

        try:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if hasattr(response, "output"):
                chunks = []
                for item in response.output:
                    if item.type == "output_text":
                        chunks.append(item.text)
                if chunks:
                    return "".join(chunks)
            if hasattr(response, "output_text"):
                return response.output_text
            return str(response)
        except Exception as exc:  # pragma: no cover - network errors
            logging.exception("LLM call failed: %s", exc)
            return self._offline_stub(prompt)

    @staticmethod
    def _offline_stub(prompt: str) -> str:
        """Return a deterministic fallback message for offline mode."""

        preview = prompt.strip().splitlines()[:3]
        summary = " ".join(preview)
        return (
            "[Offline LLM] Unable to reach the model. Prompt preview: "
            f"{summary[:500]}"
        )
