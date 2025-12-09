"""
Wrapper around the OpenAI client to capture latency and usage fields.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL

# Shared OpenAI client using the API key from config.
client = OpenAI(api_key=OPENAI_API_KEY)


def call_model(messages: List[Dict[str, str]], model: str | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Calls the chat model and returns (assistant_text, usage_dict).

    usage_dict contains:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - cached_tokens
    - latency_seconds
    """
    chosen_model = model or OPENAI_MODEL
    start = time.perf_counter()
    # prompt_cache_retention is not supported on chat.completions; rely on default caching behavior.
    response = client.chat.completions.create(
        model=chosen_model,
        messages=messages,
    )
    latency_seconds = time.perf_counter() - start

    usage = response.usage or {}
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

    # Prompt token details may be absent; default cached tokens to 0.
    prompt_details = getattr(usage, "prompt_tokens_details", None) or {}
    cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

    usage_dict: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "latency_seconds": latency_seconds,
        "cache_hit": cached_tokens > 0,
    }

    assistant_text = response.choices[0].message.content if response.choices else ""
    return assistant_text, usage_dict
