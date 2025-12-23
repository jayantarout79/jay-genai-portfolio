import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="LLM Background Mode Lab", layout="wide")


MODE_SYNC = "Synchronous (blocking)"
MODE_BG = "Background mode (polling)"
MODE_STREAM = "Background + Streaming"
DEFAULT_MODELS = ["gpt-5.2", "gpt-4.1-mini", "o3-mini"]

SESSION_DEFAULTS = {
    "active_response_id": None,
    "active_mode": None,
    "active_status": None,
    "created_at_ts": None,
    "last_poll_ts": None,
    "final_output_text": None,
    "raw_response": None,
    "stream_buffer": "",
    "stream_events": [],
    "stream_cursor": None,
    "stream_complete": False,
    "error_message": None,
    "auto_refresh_enabled": False,
    "auto_refresh_interval": 5,
    "next_auto_refresh_ts": None,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def load_env_from_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def get_client() -> OpenAI:
    load_env_from_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY not set. Please export it before using the app.")
        st.stop()
    return OpenAI(api_key=api_key)


def is_terminal(status: Optional[str]) -> bool:
    if not status:
        return False
    return status in {"completed", "failed", "cancelled", "canceled", "expired"}


def safe_dict(resp: Any) -> Dict[str, Any]:
    if resp is None:
        return {}
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "dict", "to_dict"):
        if hasattr(resp, attr):
            try:
                return getattr(resp, attr)()
            except Exception:
                pass
    if hasattr(resp, "json"):
        try:
            return json.loads(resp.json())
        except Exception:
            pass
    if hasattr(resp, "__dict__"):
        return dict(resp.__dict__)
    try:
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return {"unserializable": str(resp)}


def extract_text_from_response(resp_dict: Dict[str, Any]) -> str:
    outputs = resp_dict.get("output", []) or []
    collected = []
    for output in outputs:
        content_items = output.get("content", []) or []
        for item in content_items:
            if isinstance(item, dict):
                delta = item.get("delta") or {}
                if "text" in delta:
                    collected.append(delta["text"])
                elif "text" in item:
                    collected.append(item["text"])
    return "".join(collected)


def append_stream_text(event_dict: Dict[str, Any], buffer: str) -> str:
    outputs = event_dict.get("output", []) or []
    for output in outputs:
        for item in output.get("content", []) or []:
            if not isinstance(item, dict):
                continue
            delta = item.get("delta") or {}
            if "text" in delta:
                buffer += delta["text"]
            elif "text" in item:
                buffer += item["text"]
    return buffer


def build_input(prompt: str) -> Any:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }
    ]


def trim_events(events: List[Any], cap: int = 200) -> List[Any]:
    if len(events) <= cap:
        return events
    return events[-cap:]


def create_sync_response(prompt: str, model: str, **kwargs: Any):
    client = get_client()
    return client.responses.create(model=model, input=build_input(prompt), **kwargs)


def create_background_response(prompt: str, model: str, **kwargs: Any):
    client = get_client()
    return client.responses.create(model=model, input=build_input(prompt), background=True, **kwargs)


def create_background_stream_response(prompt: str, model: str, **kwargs: Any):
    client = get_client()
    return client.responses.create(model=model, input=build_input(prompt), background=True, stream=True, **kwargs)


def retrieve_response(resp_id: str):
    client = get_client()
    return client.responses.retrieve(resp_id)


def cancel_response(resp_id: str):
    client = get_client()
    return client.responses.cancel(resp_id)


def format_elapsed(created_at: Optional[float]) -> str:
    if not created_at:
        return "n/a"
    elapsed = max(time.time() - created_at, 0)
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    mins, secs = divmod(elapsed, 60)
    return f"{int(mins)}m {int(secs)}s"


def reset_state() -> None:
    for key in SESSION_DEFAULTS:
        st.session_state[key] = copy.deepcopy(SESSION_DEFAULTS[key])


def poll_active_response() -> None:
    resp_id = st.session_state.get("active_response_id")
    if not resp_id:
        return
    try:
        resp = retrieve_response(resp_id)
        resp_dict = safe_dict(resp)
        st.session_state.raw_response = resp_dict
        st.session_state.active_status = resp_dict.get("status")
        st.session_state.last_poll_ts = time.time()
        if not st.session_state.get("created_at_ts"):
            st.session_state.created_at_ts = resp_dict.get("created_at") or time.time()
        if is_terminal(st.session_state.active_status):
            text_out = extract_text_from_response(resp_dict)
            if text_out:
                st.session_state.final_output_text = text_out
    except Exception as exc:  # noqa: BLE001
        st.session_state.error_message = f"Retrieve failed: {exc}"


def render_status_badge(status: Optional[str]) -> None:
    if not status:
        return
    status_lower = status.lower()
    if status_lower in {"queued", "in_progress"}:
        st.info(f"Status: {status}")
    elif status_lower in {"completed"}:
        st.success(f"Status: {status}")
    elif status_lower in {"failed"}:
        st.error(f"Status: {status}")
    elif status_lower in {"cancelled", "canceled"}:
        st.warning(f"Status: {status}")
    else:
        st.write(f"Status: {status}")


init_session_state()

st.title("LLM Background Mode Lab")
st.caption("Demonstrates OpenAI Responses API across synchronous, background, and streaming modes.")

prompt_col, output_col = st.columns([0.95, 1.05])

with prompt_col:
    st.subheader("Inputs")
    prompt_text = st.text_area("Prompt", value="", height=180, placeholder="Ask the model anything...", key="prompt_input")
    model_choice = st.selectbox("Model", DEFAULT_MODELS, index=0, key="model_select")
    mode_choice = st.radio(
        "Mode",
        options=[MODE_SYNC, MODE_BG, MODE_STREAM],
        index=0,
        key="mode_select",
    )
    with st.expander("Advanced options", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_tokens_input = st.number_input(
            "Max output tokens (optional, 0 = auto)",
            min_value=0,
            value=0,
            step=64,
            key="max_output_tokens",
            help="Set to 0 to let the model decide; ignored if unsupported.",
        )
        max_tokens = int(max_tokens_input) if max_tokens_input > 0 else None
    left_buttons = st.columns(2)
    with left_buttons[0]:
        submit_clicked = st.button("Submit", type="primary", disabled=not bool(prompt_text.strip()))
    with left_buttons[1]:
        reset_clicked = st.button("Reset", type="secondary")

    if reset_clicked:
        reset_state()
        st.experimental_rerun()

with output_col:
    st.subheader("Output")

auto_refresh_container = output_col.container()
active_panel_container = output_col.container()
output_display_container = output_col.container()
raw_json_container = output_col.container()
stream_events_container = output_col.container()


def handle_submission() -> None:
    st.session_state.error_message = None
    st.session_state.final_output_text = None
    st.session_state.raw_response = None
    st.session_state.stream_buffer = ""
    st.session_state.stream_events = []
    st.session_state.stream_cursor = None
    st.session_state.stream_complete = False
    st.session_state.active_response_id = None
    st.session_state.active_status = None
    st.session_state.created_at_ts = time.time()
    st.session_state.last_poll_ts = None
    st.session_state.active_mode = mode_choice
    st.session_state.next_auto_refresh_ts = None

    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_tokens:
        kwargs["max_output_tokens"] = int(max_tokens)

    if mode_choice == MODE_SYNC:
        try:
            resp = create_sync_response(prompt_text, model_choice, **kwargs)
            resp_dict = safe_dict(resp)
            st.session_state.active_response_id = resp_dict.get("id")
            st.session_state.active_status = resp_dict.get("status", "completed")
            st.session_state.raw_response = resp_dict
            st.session_state.created_at_ts = resp_dict.get("created_at") or st.session_state.created_at_ts
            st.session_state.final_output_text = extract_text_from_response(resp_dict)
            if not st.session_state.active_status:
                st.session_state.active_status = "completed"
        except Exception as exc:  # noqa: BLE001
            st.session_state.error_message = f"Sync request failed: {exc}"
    elif mode_choice == MODE_BG:
        try:
            resp = create_background_response(prompt_text, model_choice, **kwargs)
            resp_dict = safe_dict(resp)
            st.session_state.active_response_id = resp_dict.get("id")
            st.session_state.active_status = resp_dict.get("status", "queued")
            st.session_state.raw_response = resp_dict
            st.session_state.created_at_ts = resp_dict.get("created_at") or st.session_state.created_at_ts
        except Exception as exc:  # noqa: BLE001
            st.session_state.error_message = f"Background request failed: {exc}"
    else:
        output_placeholder = output_display_container.empty()
        st.session_state.active_status = "in_progress"
        try:
            resp_stream = create_background_stream_response(prompt_text, model_choice, **kwargs)
            last_event_dict: Optional[Dict[str, Any]] = None
            for event in resp_stream:
                event_dict = safe_dict(event)
                last_event_dict = event_dict
                resp_meta = event_dict.get("response") or {}
                candidate_id = event_dict.get("id") or resp_meta.get("id")
                if not st.session_state.active_response_id and candidate_id:
                    st.session_state.active_response_id = candidate_id
                if event_dict.get("status"):
                    st.session_state.active_status = event_dict.get("status")
                if resp_meta.get("status"):
                    st.session_state.active_status = resp_meta.get("status")
                if event_dict.get("created_at") and not st.session_state.created_at_ts:
                    st.session_state.created_at_ts = event_dict.get("created_at")
                if resp_meta.get("created_at") and not st.session_state.created_at_ts:
                    st.session_state.created_at_ts = resp_meta.get("created_at")
                st.session_state.stream_cursor = event_dict.get("sequence_number", st.session_state.stream_cursor)
                st.session_state.stream_buffer = append_stream_text(event_dict, st.session_state.stream_buffer)
                st.session_state.stream_events = trim_events(
                    st.session_state.stream_events
                    + [
                        {
                            "type": event_dict.get("type") or resp_meta.get("status"),
                            "cursor": event_dict.get("sequence_number"),
                            "delta": append_stream_text(event_dict, ""),
                            "status": event_dict.get("status") or resp_meta.get("status"),
                        }
                    ]
                )
                output_placeholder.markdown(st.session_state.stream_buffer or "_Streaming..._")
            st.session_state.stream_complete = True
            st.session_state.active_status = st.session_state.active_status or "in_progress"
            st.session_state.final_output_text = st.session_state.stream_buffer
            final_id = (
                (last_event_dict or {}).get("id")
                or (last_event_dict or {}).get("response", {}).get("id")
                or st.session_state.get("active_response_id")
            )
            if final_id:
                try:
                    resp = retrieve_response(final_id)
                    resp_dict = safe_dict(resp)
                    st.session_state.raw_response = resp_dict
                    st.session_state.active_response_id = resp_dict.get("id") or st.session_state.active_response_id
                    st.session_state.active_status = resp_dict.get("status") or st.session_state.active_status
                    st.session_state.created_at_ts = resp_dict.get("created_at") or st.session_state.created_at_ts
                except Exception:
                    pass
            else:
                st.session_state.raw_response = last_event_dict
            if st.session_state.stream_complete and not is_terminal(st.session_state.active_status):
                st.session_state.active_status = "completed"
        except Exception as exc:  # noqa: BLE001
            st.session_state.error_message = f"Streaming request failed: {exc}"


if submit_clicked:
    handle_submission()


def render_active_panel() -> None:
    status_val = st.session_state.get("active_status")
    if not st.session_state.get("active_response_id") or is_terminal(status_val):
        return
    with active_panel_container:
        with st.container(border=True):
            st.write("**Active job**")
            st.write(f"ID: `{st.session_state.active_response_id}`")
            render_status_badge(status_val)
            st.write(f"Created: {format_elapsed(st.session_state.created_at_ts)} ago")
            if st.session_state.last_poll_ts:
                last_poll = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_poll_ts))
                st.caption(f"Last poll: {last_poll}")
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("Refresh status", key="refresh_status_btn"):
                    poll_active_response()
            with btn_cols[1]:
                status_lower = status_val.lower() if status_val else ""
                if status_lower in {"queued", "in_progress"}:
                    if st.button("Cancel job", key="cancel_job_btn"):
                        try:
                            resp = cancel_response(st.session_state.active_response_id)
                            resp_dict = safe_dict(resp)
                            st.session_state.raw_response = resp_dict
                            st.session_state.active_status = resp_dict.get("status", "cancelled")
                        except Exception as exc:  # noqa: BLE001
                            st.session_state.error_message = f"Cancel failed: {exc}"
            with btn_cols[2]:
                st.write("")
            if st.session_state.active_mode == MODE_STREAM:
                st.caption(
                    f"Cursor: {st.session_state.stream_cursor} | Stream complete: {st.session_state.stream_complete}"
                )
            if status_val and not is_terminal(status_val):
                st.checkbox(
                    "Auto-refresh",
                    value=st.session_state.auto_refresh_enabled,
                    key="auto_refresh_enabled",
                )
                st.slider(
                    "Interval (seconds)",
                    2,
                    10,
                    st.session_state.auto_refresh_interval,
                    key="auto_refresh_interval",
                )
            else:
                st.session_state.auto_refresh_enabled = False
                st.session_state.next_auto_refresh_ts = None


render_active_panel()


def maybe_auto_refresh() -> None:
    status_val = st.session_state.get("active_status")
    if (
        st.session_state.get("active_response_id")
        and status_val
        and not is_terminal(status_val)
        and st.session_state.get("auto_refresh_enabled")
    ):
        interval = st.session_state.get("auto_refresh_interval", 5)
        auto_func = getattr(st, "autorefresh", None)
        if callable(auto_func):
            auto_func(interval=interval * 1000, key="auto_refresh_tick")
            poll_active_response()
        else:
            now = time.time()
            next_ts = st.session_state.get("next_auto_refresh_ts")
            if not next_ts:
                st.session_state.next_auto_refresh_ts = now + interval
            if now >= (st.session_state.next_auto_refresh_ts or 0):
                poll_active_response()
                st.session_state.next_auto_refresh_ts = now + interval
                st.experimental_rerun()


maybe_auto_refresh()


def render_output_panel() -> None:
    with output_display_container:
        mode_label = st.session_state.get("active_mode") or mode_choice
        st.write(f"**Current mode:** {mode_label}")
        st.write(f"Response ID: `{st.session_state.get('active_response_id') or 'n/a'}`")
        render_status_badge(st.session_state.get("active_status"))
        st.write(f"Elapsed since creation: {format_elapsed(st.session_state.get('created_at_ts'))}")
        if st.session_state.error_message:
            st.error(st.session_state.error_message)
        output_text = st.session_state.final_output_text or st.session_state.stream_buffer
        if output_text:
            st.markdown(output_text)
        else:
            st.info("No output yet. Submit a prompt or refresh an active job.")


render_output_panel()


def render_raw_and_events() -> None:
    with raw_json_container:
        with st.expander("Raw response JSON", expanded=False):
            if st.session_state.raw_response:
                st.json(st.session_state.raw_response)
            else:
                st.write("No response data yet.")
    with stream_events_container:
        with st.expander("Stream event log", expanded=False):
            if st.session_state.stream_events:
                for ev in st.session_state.stream_events:
                    st.write(ev)
            else:
                st.write("No stream events captured.")
        if st.session_state.active_mode == MODE_STREAM and not st.session_state.stream_complete:
            if st.button("Reconnect (resume)", key="reconnect_btn"):
                st.warning(
                    "SDK resume streaming not supported in this environment yet; use polling to retrieve final output."
                )


render_raw_and_events()

with output_col:
    st.caption("Notes: background responses may expire after ~10 minutes. ZDR not supported in this sample.")


# How to run
# pip install streamlit openai
# export OPENAI_API_KEY=...
# streamlit run app.py
