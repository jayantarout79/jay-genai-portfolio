"""Minimal D-ID talks client."""

from __future__ import annotations

from typing import Dict, Optional

import requests


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def create_talk(api_key: str, script: str, source_url: str) -> Optional[str]:
    """Create a talk and return its id."""
    if not api_key or not script or not source_url:
        return None
    payload = {
        "source_url": source_url,
        "script": {"type": "text", "input": script},
    }
    response = requests.post("https://api.d-id.com/talks", json=payload, headers=_headers(api_key))
    response.raise_for_status()
    return response.json().get("id")


def fetch_talk(api_key: str, talk_id: str) -> Dict:
    """Fetch talk status/details."""
    if not api_key or not talk_id:
        return {}
    response = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=_headers(api_key))
    response.raise_for_status()
    return response.json() or {}
