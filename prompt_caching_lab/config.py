"""
Configuration loader for environment variables and pricing constants.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    """Fetch an environment variable, optionally falling back to a default."""
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(
            f"Environment variable {name} is required. "
            "Create a .env file or export it before running the app."
        )
    return value


OPENAI_API_KEY: str = _get_env("OPENAI_API_KEY", None)
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.1")


@dataclass(frozen=True)
class PriceSheet:
    """Approximate token pricing per 1K tokens; update as needed."""

    prompt_per_1k: float
    cached_prompt_per_1k: float
    completion_per_1k: float


def get_price_sheet(model: str) -> PriceSheet:
    """
    Return example pricing per 1K tokens for supported models.

    Prices below are scaled from OpenAI's published per‑million token rates.
    """
    price_map: dict[str, PriceSheet] = {
        # 1.25 / 1000, 0.125 / 1000, 10 / 1000
        "gpt-5.1": PriceSheet(0.00125, 0.000125, 0.01000),
        "gpt-5.1-chat-latest": PriceSheet(0.00125, 0.000125, 0.01000),
        "gpt-5.1-codex-max": PriceSheet(0.00125, 0.000125, 0.01000),
        "gpt-5.1-codex": PriceSheet(0.00125, 0.000125, 0.01000),
        # 1.25 / 1000, 0.125 / 1000, 10 / 1000
        "gpt-5": PriceSheet(0.00125, 0.000125, 0.01000),
        "gpt-5-chat-latest": PriceSheet(0.00125, 0.000125, 0.01000),
        "gpt-5-codex": PriceSheet(0.00125, 0.000125, 0.01000),
        # 0.25 / 1000, 0.025 / 1000, 2 / 1000
        "gpt-5-mini": PriceSheet(0.00025, 0.000025, 0.00200),
        # 0.05 / 1000, 0.005 / 1000, 0.4 / 1000
        "gpt-5-nano": PriceSheet(0.00005, 0.000005, 0.00040),
        # 15 / 1000, no cached rate listed, 120 / 1000
        "gpt-5-pro": PriceSheet(0.01500, 0.00000, 0.12000),
        # 2 / 1000, 0.5 / 1000, 8 / 1000
        "gpt-4.1": PriceSheet(0.00200, 0.00050, 0.00800),
        # 0.4 / 1000, 0.1 / 1000, 1.6 / 1000
        "gpt-4.1-mini": PriceSheet(0.00040, 0.00010, 0.00160),
        # 0.1 / 1000, 0.025 / 1000, 0.4 / 1000
        "gpt-4.1-nano": PriceSheet(0.00010, 0.000025, 0.00040),
        # 2.5 / 1000, 1.25 / 1000, 10 / 1000
        "gpt-4o": PriceSheet(0.00250, 0.00125, 0.01000),
    }
    return price_map.get(model, price_map["gpt-5.1"])
