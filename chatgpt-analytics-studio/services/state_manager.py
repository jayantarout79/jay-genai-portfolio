import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

STATE_PATH = os.path.join("data", "processing_state.json")


def load_state() -> Dict[str, Any]:
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def ensure_state_basics(raw_folder_path: str, raw_fingerprint: Optional[str]) -> Dict[str, Any]:
    state = load_state()
    state["raw_folder_path"] = raw_folder_path
    state["raw_fingerprint"] = raw_fingerprint
    state["last_checked"] = datetime.utcnow().isoformat() + "Z"
    save_state(state)
    return state


def fingerprint_matches(state: Dict[str, Any], fingerprint: Optional[str]) -> bool:
    return fingerprint is not None and state.get("raw_fingerprint") == fingerprint
