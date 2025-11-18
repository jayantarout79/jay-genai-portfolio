"""Configuration helpers for the Customer Experience Analyzer.

This module focuses on safe loading of environment variables and
centralizing constants that control application behavior so that the
rest of the codebase can depend on a single source of truth."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


@dataclass(frozen=True)
class AppConfig:
    """Holds runtime configuration for the Streamlit app."""

    openai_api_key: Optional[str]
    openai_model: str
    default_theme_count: int = 6
    max_example_comments: int = 2
    trend_frequency: str = "W"  # weekly trend by default


def _load_env_file(env_path: str | Path = ".env") -> None:
    """Load environment variables from a .env file if present."""

    load_dotenv(dotenv_path=env_path, override=False)


def load_config(api_key_override: Optional[str] = None) -> AppConfig:
    """Return the active application configuration.

    Parameters
    ----------
    api_key_override:
        Optional API key supplied via the UI. If provided, it takes
        precedence over the value loaded from the environment file.
    """

    _load_env_file()
    api_key = api_key_override or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return AppConfig(openai_api_key=api_key, openai_model=model_name)
