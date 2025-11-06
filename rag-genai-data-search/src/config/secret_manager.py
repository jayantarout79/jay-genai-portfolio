from __future__ import annotations
import os
import stat
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv, set_key, get_key

DEFAULT_ENV_PATH = Path(".env")

def ensure_env_file(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Ensure the .env file exists; create with safe permissions on POSIX."""
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("", encoding="utf-8")
    try:
        if os.name == "posix":
            env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except Exception:
        pass

def load_env(env_path: Path = DEFAULT_ENV_PATH, override: bool = False) -> None:
    load_dotenv(dotenv_path=env_path, override=override)

def get_env_value(key: str, default: Optional[str] = None, env_path: Path = DEFAULT_ENV_PATH) -> Optional[str]:
    load_env(env_path=env_path, override=False)
    val = get_key(str(env_path), key)
    return val if val is not None else default

def set_env_value(key: str, value: str, env_path: Path = DEFAULT_ENV_PATH) -> None:
    ensure_env_file(env_path)
    set_key(str(env_path), key, value)

def set_env_values(pairs: Dict[str, str], env_path: Path = DEFAULT_ENV_PATH) -> None:
    ensure_env_file(env_path)
    for k, v in pairs.items():
        set_key(str(env_path), k, v)