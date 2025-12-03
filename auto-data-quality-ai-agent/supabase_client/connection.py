from __future__ import annotations

from supabase import create_client, Client

from config import get_settings, Settings


def get_client(settings: Settings | None = None) -> Client:
    """
    Create a Supabase client using provided settings or environment-based settings.
    """
    settings = settings or get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)
