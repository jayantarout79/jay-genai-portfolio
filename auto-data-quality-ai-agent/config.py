from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file (project root) if present for developer convenience.
# override=True ensures the .env values replace any placeholder values from other sources.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


def _get_secret(name: str) -> Optional[str]:
    """
    Read a secret from environment or Streamlit secrets (if available).
    """
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st  # type: ignore

        value = st.secrets.get(name)  # type: ignore[arg-type]
        return value
    except Exception:
        return None


@dataclass
class Settings:
    supabase_url: str
    supabase_key: str
    supabase_anon_key: Optional[str] = None


def get_settings() -> Settings:
    """
    Load Supabase settings from environment variables or Streamlit secrets.

    Raises:
        KeyError: If any required environment variable is missing.
    """
    url = _get_secret("SUPABASE_URL") or _get_secret("NEXT_PUBLIC_SUPABASE_URL")
    key = (
        _get_secret("SUPABASE_SERVICE_ROLE_KEY")
        or _get_secret("SUPABASE_KEY")
        or _get_secret("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    )
    anon_key = _get_secret("NEXT_PUBLIC_SUPABASE_ANON_KEY")

    missing = [name for name, value in [
        ("SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL", url),
        ("SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY/NEXT_PUBLIC_SUPABASE_ANON_KEY", key),
    ] if not value]
    if missing:
        raise KeyError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        supabase_url=url,
        supabase_key=key,
        supabase_anon_key=anon_key,
    )
