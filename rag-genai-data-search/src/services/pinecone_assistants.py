from __future__ import annotations
from typing import Optional as _Opt, Dict as _Dict, Tuple as _Tuple, List as _List, Any as _Any

# -----------------------------
# Normalizers & status mappers
# -----------------------------

def _normalize_assistant_items(res: _Any) -> _List[dict]:
    """Normalize various SDK response shapes into a simple list of dict-like items."""
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        for k in ("assistants", "data", "items"):
            if isinstance(res.get(k), list):
                return res[k]
        return []
    for attr in ("assistants", "data", "items"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            if isinstance(val, list):
                return val
    return []


def status_icon_for(value: str) -> str:
    """Map assistant status to a colored icon."""
    if not value:
        return "⚪"
    v = value.lower()
    if v in {"ready", "active", "enabled", "available"}:
        return "🟢"
    if v in {"building", "indexing", "creating", "pending", "processing"}:
        return "🟡"
    if v in {"error", "failed", "disabled"}:
        return "🔴"
    return "⚪"


def file_status_icon(value: str) -> str:
    """Map file status to a colored icon."""
    if not value:
        return "⚪"
    v = value.lower()
    if v in {"available", "ready"}:
        return "🟢"
    if v in {"processing", "indexing", "pending"}:
        return "🟡"
    if v in {"error", "failed"}:
        return "🔴"
    return "⚪"

# -----------------------------
# Assistants: list / describe
# -----------------------------

def list_pinecone_assistants(api_key: str) -> _Tuple[_List[dict], str | None]:
    """Return (items, error). If error is not None, items will be empty."""
    try:
        from pinecone import Pinecone
    except Exception:
        return [], "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"
    try:
        pc = Pinecone(api_key=api_key)
        res = pc.assistant.list_assistants()
        items = _normalize_assistant_items(res)
        return items, None
    except Exception as e:
        return [], f"Error listing assistants: {e}"


def describe_pinecone_assistant(api_key: str, name_or_id: str) -> _Tuple[dict | None, str | None]:
    """Return (assistant_dict, error). Uses assistant_name for compatibility."""
    try:
        from pinecone import Pinecone
    except Exception:
        return None, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"
    try:
        pc = Pinecone(api_key=api_key)
        res = pc.assistant.describe_assistant(assistant_name=name_or_id)
        if isinstance(res, dict):
            return res, None
        out = {k: getattr(res, k) for k in dir(res) if not k.startswith("_") and not callable(getattr(res, k))}
        return out, None
    except Exception as e:
        return None, f"Error describing assistant '{name_or_id}': {e}"

# -----------------------------
# Assistants: create / delete
# -----------------------------

def create_pinecone_assistant(
    api_key: str,
    assistant_name: str,
    instructions: _Opt[str] = None,
    region: str = "us",
    timeout: int = 30,
) -> _Tuple[dict | None, str | None]:
    """Create a new Pinecone assistant. Returns (assistant_dict, error)."""
    try:
        from pinecone import Pinecone
    except Exception:
        return None, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return None, "assistant_name cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        payload: dict = {
            "assistant_name": assistant_name.strip(),
            "region": region,
            "timeout": timeout,
        }
        if instructions and instructions.strip():
            payload["instructions"] = instructions.strip()
        res = pc.assistant.create_assistant(**payload)
        if isinstance(res, dict):
            return res, None
        out = {k: getattr(res, k) for k in dir(res) if not k.startswith("_") and not callable(getattr(res, k))}
        return out, None
    except Exception as e:
        return None, f"Error creating assistant '{assistant_name}': {e}"


def delete_pinecone_assistant(api_key: str, name_or_id: str) -> _Tuple[bool, str | None]:
    """Delete a Pinecone assistant by name/id. Returns (ok, error)."""
    try:
        from pinecone import Pinecone
    except Exception:
        return False, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not name_or_id or not name_or_id.strip():
        return False, "assistant_name cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        pc.assistant.delete_assistant(assistant_name=name_or_id.strip())
        return True, None
    except Exception as e:
        return False, f"Error deleting assistant '{name_or_id}': {e}"

# -----------------------------
# Files: list / describe / delete / upload
# -----------------------------

def list_assistant_files(
    api_key: str,
    assistant_name: str,
    metadata_filter: _Opt[_Dict[str, object]] = None,
) -> _Tuple[_List[dict], str | None]:
    """List files attached to an assistant. Returns (files, error)."""
    try:
        from pinecone import Pinecone
    except Exception:
        return [], "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return [], "assistant_name cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        asst = pc.assistant.Assistant(assistant_name=assistant_name.strip())
        if metadata_filter:
            res = asst.list_files(filter=metadata_filter)
        else:
            res = asst.list_files()
        if isinstance(res, dict) and isinstance(res.get("files"), list):
            return res["files"], None
        if isinstance(res, list):
            return res, None
        if hasattr(res, "files") and isinstance(res.files, list):
            return res.files, None
        return [], None
    except Exception as e:
        return [], f"Error listing files for assistant '{assistant_name}': {e}"


def describe_assistant_file(
    api_key: str,
    assistant_name: str,
    file_id: str,
    include_url: bool = False,
) -> _Tuple[dict | None, str | None]:
    """Describe a specific file for an assistant. Returns (file_dict, error)."""
    try:
        from pinecone import Pinecone
    except Exception:
        return None, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return None, "assistant_name cannot be empty"
    if not file_id or not file_id.strip():
        return None, "file_id cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        asst = pc.assistant.Assistant(assistant_name=assistant_name.strip())
        res = asst.describe_file(file_id=file_id.strip(), include_url=include_url)
        if isinstance(res, dict):
            return res, None
        out = {k: getattr(res, k) for k in dir(res) if not k.startswith("_") and not callable(getattr(res, k))}
        return out, None
    except Exception as e:
        return None, f"Error describing file '{file_id}' for '{assistant_name}': {e}"


def delete_assistant_file(api_key: str, assistant_name: str, file_id: str) -> _Tuple[bool, str | None]:
    """Delete a specific file from an assistant. Returns (ok, error)."""
    try:
        from pinecone import Pinecone
    except Exception:
        return False, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return False, "assistant_name cannot be empty"
    if not file_id or not file_id.strip():
        return False, "file_id cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        asst = pc.assistant.Assistant(assistant_name=assistant_name.strip())
        asst.delete_file(file_id=file_id.strip())
        return True, None
    except Exception as e:
        return False, f"Error deleting file '{file_id}' from '{assistant_name}': {e}"


def upload_file_to_assistant(
    api_key: str,
    assistant_name: str,
    file_path: str,
    metadata: _Opt[_Dict[str, object]] = None,
) -> _Tuple[dict | None, str | None]:
    """Upload a local file to an assistant. Returns (file_dict, error).

    We avoid `pc.files` to support SDKs without that attribute. Strategy:
      1) Prefer the Assistant instance method: `asst.upload_file(file_path=..., metadata=...)`.
      2) Fallback to namespaced method: `pc.assistant.upload_file(assistant_name=..., file_path=..., metadata=...)`.
    """
    try:
        from pinecone import Pinecone
    except Exception:
        return None, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return None, "assistant_name cannot be empty"
    if not file_path or not str(file_path).strip():
        return None, "file_path cannot be empty"

    try:
        pc = Pinecone(api_key=api_key)
        asst = pc.assistant.Assistant(assistant_name=assistant_name.strip())

        # Primary path: Assistant instance method (note: file_path kw)
        try:
            if hasattr(asst, "upload_file"):
                kwargs = {"file_path": file_path}
                if metadata:
                    kwargs["metadata"] = metadata
                res = asst.upload_file(**kwargs)
                if isinstance(res, dict):
                    return res, None
                out = {k: getattr(res, k) for k in dir(res) if not k.startswith("_") and not callable(getattr(res, k))}
                return out, None
        except Exception as e:
            first_err = f"Assistant.upload_file failed: {e}"
        else:
            first_err = None

        # Fallback path: namespaced helper under pc.assistant
        try:
            if hasattr(pc.assistant, "upload_file"):
                kwargs = {"assistant_name": assistant_name.strip(), "file_path": file_path}
                if metadata:
                    kwargs["metadata"] = metadata
                res = pc.assistant.upload_file(**kwargs)
                if isinstance(res, dict):
                    return res, None
                out = {k: getattr(res, k) for k in dir(res) if not k.startswith("_") and not callable(getattr(res, k))}
                return out, None
        except Exception as e:
            second_err = f"pc.assistant.upload_file failed: {e}"
        else:
            second_err = None

        # If both paths failed, return combined error info
        errs = "; ".join([e for e in [first_err, second_err] if e]) or "No supported upload method found in this SDK version"
        return None, errs

    except Exception as e:
        return None, f"Error uploading '{file_path}' to '{assistant_name}': {e}"

# -----------------------------
# Chat: standard interface
# -----------------------------

def _to_chat_messages(raw_msgs):
    """Convert list of {role, content} dicts to SDK Message objects when available."""
    msgs = []
    try:
        from pinecone_plugins.assistant.models.chat import Message  # type: ignore
        for m in raw_msgs:
            if isinstance(m, dict):
                msgs.append(Message(role=m.get("role"), content=m.get("content")))
            else:
                msgs.append(m)
    except Exception:
        # Fallback: just pass dicts through
        msgs = raw_msgs
    return msgs


def chat_with_assistant(
    api_key: str,
    assistant_name: str,
    messages: list[dict],
    *,
    model: _Opt[str] = None,
    json_response: bool = False,
    stream: bool = False,
    include_highlights: bool = False,
    filter: _Opt[_Dict[str, object]] = None,
    context_options: _Opt[_Dict[str, object]] = None,
    temperature: _Opt[float] = None,
) -> tuple[dict | list | None, str | None]:
    """Chat using Pinecone's standard interface. Returns (response, error).

    Default is a non-streaming structured response (recommended for Streamlit UI).
    """
    try:
        from pinecone import Pinecone
    except Exception:
        return None, "Missing dependency. Run: pip install --upgrade pinecone pinecone-plugin-assistant"

    if not assistant_name or not assistant_name.strip():
        return None, "assistant_name cannot be empty"
    if not messages or not isinstance(messages, list):
        return None, "messages must be a non-empty list"

    try:
        pc = Pinecone(api_key=api_key)
        asst = pc.assistant.Assistant(assistant_name=assistant_name.strip())

        kwargs: dict = {"messages": _to_chat_messages(messages)}
        if model:
            kwargs["model"] = model
        if json_response:
            kwargs["json_response"] = True
        if stream:
            kwargs["stream"] = True
        if include_highlights:
            kwargs["include_highlights"] = True
        if filter:
            kwargs["filter"] = filter
        if context_options:
            kwargs["context_options"] = context_options
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        resp = asst.chat(**kwargs)

        if stream:
            # In streaming mode the SDK yields chunks; return as-is
            return resp, None
        if isinstance(resp, dict):
            return resp, None

        # Normalize non-dict SDK object
        out = {k: getattr(resp, k) for k in dir(resp)
               if not k.startswith("_") and not callable(getattr(resp, k))}
        return out, None

    except Exception as e:
        return None, f"Error chatting with assistant '{assistant_name}': {e}"