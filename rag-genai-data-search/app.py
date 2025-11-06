import os
import tempfile  # used for handling uploads safely if needed
from typing import Optional as _Opt
from pathlib import Path
import streamlit as st

# --- make imports robust regardless of where Streamlit is launched from
import sys
BASE_DIR = Path(__file__).resolve().parent
for p in [BASE_DIR, BASE_DIR / "src", BASE_DIR.parent]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

# Local helper to manage .env
try:
    from src.config.secret_manager import (
        ensure_env_file,
        get_env_value,
        set_env_value,
        set_env_values,
    )
except ModuleNotFoundError:
    from config.secret_manager import (
        ensure_env_file,
        get_env_value,
        set_env_value,
        set_env_values,
    )

# Pinecone assistants service (modularized with robust fallback)
IMPORT_ERR = None
try:
    from src.services.pinecone_assistants import (
        list_pinecone_assistants,
        describe_pinecone_assistant,
        status_icon_for,
        create_pinecone_assistant,
        delete_pinecone_assistant,
        list_assistant_files,
        describe_assistant_file,
        delete_assistant_file,
        upload_file_to_assistant,
        file_status_icon,
        chat_with_assistant,
    )
except Exception as e1:
    try:
        from services.pinecone_assistants import (
            list_pinecone_assistants,
            describe_pinecone_assistant,
            status_icon_for,
            create_pinecone_assistant,
            delete_pinecone_assistant,
            list_assistant_files,
            describe_assistant_file,
            delete_assistant_file,
            upload_file_to_assistant,
            file_status_icon,
            chat_with_assistant,
        )
    except Exception as e2:
        IMPORT_ERR = f"Failed to import pinecone_assistants module. Errors:\n- {e1}\n- {e2}"
        list_pinecone_assistants = None
        describe_pinecone_assistant = None
        status_icon_for = None
        create_pinecone_assistant = None
        delete_pinecone_assistant = None
        list_assistant_files = None
        describe_assistant_file = None
        delete_assistant_file = None
        upload_file_to_assistant = None
        file_status_icon = None
        chat_with_assistant = None

st.set_page_config(page_title="Pinecone RAG Dashboard — Settings", layout="wide")

# --- UI Header

# --- Where the .env lives (project root by default)
ENV_PATH = Path(".env")

# Make sure .env exists with safe permissions on POSIX
ensure_env_file(ENV_PATH)

# Keep show/hide state for the form
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

# Track selected assistant
if "selected_assistant" not in st.session_state:
    st.session_state.selected_assistant = None  # store name/id
if "selected_assistant_title" not in st.session_state:
    st.session_state.selected_assistant_title = None

# Upload panel visibility toggle
if "upload_mode" not in st.session_state:
    st.session_state.upload_mode = True  # allow uploads by default

# Chat UI state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# --- Session-only Pinecone key (no persistence)
if "pinecone_api_key" not in st.session_state:
    st.session_state.pinecone_api_key = None

def active_key():
    """Return the active Pinecone key for this session only.
    Falls back to Streamlit secrets or .env for local/dev convenience,
    but the Settings UI will not persist anything to disk.
    """
    if st.session_state.pinecone_api_key:
        return st.session_state.pinecone_api_key
    try:
        if hasattr(st, "secrets") and "PINECONE_API_KEY" in st.secrets:
            return st.secrets["PINECONE_API_KEY"]
    except Exception:
        pass
    return get_env_value("PINECONE_API_KEY", env_path=ENV_PATH)

# Current value (masked)
current_key = active_key()

# Sidebar + Main toggle
sidebar_clicked = False
with st.sidebar:
    if st.button("Open Settings", use_container_width=True, key="open_settings_sidebar"):
        sidebar_clicked = True

    st.markdown("---")
    st.subheader("Pinecone Assistants")
    if IMPORT_ERR:
        st.error(IMPORT_ERR)
    refresh_assistants = st.button("🔄 Refresh Assistants", use_container_width=True, key="refresh_assistants")

    if current_key and list_pinecone_assistants is not None:
        if refresh_assistants:
            st.session_state["refresh_trigger"] = True
            st.rerun()

        items, err = list_pinecone_assistants(current_key)
        if err:
            st.warning(err)
        else:
            # Build options for selection
            options = []  # list of (label, id)
            id_to_title = {}
            if items:
                for a in items:
                    if isinstance(a, dict):
                        name = a.get("name") or a.get("display_name") or a.get("id")
                        _id = a.get("id") or a.get("assistant_id") or name
                    else:
                        name = getattr(a, "name", None) or getattr(a, "display_name", None) or getattr(a, "id", None)
                        _id = getattr(a, "id", None) or getattr(a, "assistant_id", None) or name

                    # Light label (avoid per-item describe to reduce API traffic)
                    icon = "🗂️"
                    title = name or "(unnamed assistant)"
                    label = f"{icon} {title}"
                    options.append((_id, label))
                    id_to_title[_id] = title

                # Inject a "No selection" option at the top
                option_ids = ["__NONE__"] + [o[0] for o in options]
                option_labels = {"__NONE__": "— No selection —"}
                option_labels.update({oid: lbl for oid, lbl in options})

                # Determine current index
                try:
                    current_idx = option_ids.index(st.session_state.selected_assistant) if st.session_state.selected_assistant in option_ids else 0
                except Exception:
                    current_idx = 0

                selected = st.radio(
                    label="Select Assistant",
                    options=option_ids,
                    index=current_idx,
                    format_func=lambda oid: option_labels.get(oid, oid),
                    key="assistant_radio",
                ) if option_ids else None

                if selected == "__NONE__":
                    st.session_state.selected_assistant = None
                    st.session_state.selected_assistant_title = None
                elif selected:
                    st.session_state.selected_assistant = selected
                    st.session_state.selected_assistant_title = id_to_title.get(selected, selected)

                # Clear selection control
                if st.button("Clear selection", key="clear_asst_selection"):
                    st.session_state.selected_assistant = None
                    st.session_state.selected_assistant_title = None
                    st.rerun()

                # Toggle upload panel visibility
                st.checkbox(
                    "Enable upload panel",
                    value=st.session_state.upload_mode,
                    key="upload_mode",
                    help="Uncheck to hide the Upload section in the Files panel.",
                )

                # Open chat panel when an assistant is selected
                can_chat = bool(st.session_state.selected_assistant)
                if st.button("💬 Ask Assistant", disabled=not can_chat, use_container_width=True, key="open_chat_btn"):
                    st.session_state.chat_open = True
                    st.session_state.show_settings = False
                    if st.session_state.selected_assistant not in st.session_state.chat_history:
                        st.session_state.chat_history[st.session_state.selected_assistant] = []
                    st.rerun()
                # Switch back to Files view
                if st.button("📄 Manage Files", disabled=not can_chat, use_container_width=True, key="open_files_btn"):
                    st.session_state.chat_open = False
                    st.session_state.show_settings = False
                    st.rerun()

                # Per-assistant delete buttons
                if options:
                    st.caption("Manage assistants")
                    for oid, label in options:
                        cols = st.columns([5, 1])
                        with cols[0]:
                            st.write(label)
                        with cols[1]:
                            if delete_pinecone_assistant is not None and st.button("🗑️", key=f"del_asst_{oid}"):
                                ok, derr = delete_pinecone_assistant(current_key, oid)
                                if ok:
                                    st.success("Deleted assistant.")
                                else:
                                    st.error(derr or "Failed to delete.")
                                st.rerun()
            else:
                st.info("No assistants found.")

        # Create-assistant form at bottom
        st.markdown("---")
        st.caption("Create a new assistant")
        with st.form("create_assistant_form"):
            new_name = st.text_input("Name", key="new_asst_name")
            new_instr = st.text_area("Instructions (optional)", key="new_asst_instr", height=80)
            colr, colt = st.columns([1,1])
            with colr:
                new_region = st.selectbox("Region", options=["us", "eu"], index=0, key="new_asst_region")
            with colt:
                new_timeout = st.number_input("Timeout (s)", min_value=5, max_value=120, value=30, step=5, key="new_asst_timeout")
            create_clicked = st.form_submit_button("Create Assistant", disabled=(create_pinecone_assistant is None))
        if create_clicked:
            if not new_name or not new_name.strip():
                st.error("Assistant name is required.")
            else:
                res, cerr = create_pinecone_assistant(current_key, new_name.strip(), new_instr, new_region, int(new_timeout)) if create_pinecone_assistant else (None, "Create helper missing")
                if cerr:
                    st.error(cerr)
                else:
                    st.success(f"Assistant '{new_name.strip()}' created.")
                    st.rerun()

# Only sidebar controls settings visibility
if sidebar_clicked:
    st.session_state.show_settings = True

# Auto-open settings if the API key is not set (first-time UX)
if not current_key and not st.session_state.show_settings:
    st.session_state.show_settings = True

# Main area (visible only when settings are open)
if st.session_state.show_settings:
    st.title("🔧 Settings")
    st.caption("Add your Pinecone API key for **this session only**. It is not saved to disk or secrets.")
    with st.container(border=True):
        st.subheader("Pinecone Configuration")

        # Small status line
        status = "✅ Set (session)" if st.session_state.pinecone_api_key else ("✅ Found (fallback)" if current_key else "❌ Not set")
        st.write(f"**PINECONE_API_KEY:** {status}")
        st.caption("Session mode: paste your key below and click ‘Use for this session’. It will be cleared when the session ends.")

        st.divider()

        # Settings form
        with st.expander("Edit Settings", expanded=True):
            with st.form("pinecone_settings_form", clear_on_submit=False):
                api_key_input = st.text_input(
                    "Pinecone API Key (session-only)",
                    value=st.session_state.pinecone_api_key or "",
                    type="password",
                    help="Used only for this session. Not persisted.",
                )

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    use_session = st.form_submit_button("Use for this session", type="primary")
                with col2:
                    clear_key = st.form_submit_button("Clear key")
                with col3:
                    close = st.form_submit_button("Close Settings")

            if use_session:
                if not api_key_input or not api_key_input.strip():
                    st.error("Pinecone API key cannot be empty.")
                else:
                    if not api_key_input.startswith("pc-"):
                        st.warning("The key doesn’t start with `pc-`. Double-check you pasted the correct Pinecone API key.")
                    st.session_state.pinecone_api_key = api_key_input.strip()
                    st.success("Session key set. You can now use assistants.")
                    st.session_state.show_settings = False
                    st.rerun()

            if clear_key:
                st.session_state.pinecone_api_key = None
                st.info("Session key cleared.")
                st.rerun()

            if 'close' in locals() and close:
                st.session_state.show_settings = False
                st.info("Settings closed.")
                st.rerun()

# Re-resolve key after settings changes
current_key = active_key()

# -------- Assistant Files panel (visible when an assistant is selected and settings are closed) --------
if (
    not st.session_state.show_settings
    and current_key
    and st.session_state.selected_assistant
    and list_assistant_files is not None
    and not st.session_state.chat_open  # hide Files when Chat is open
):
    with st.container(border=True):
        asst_name = st.session_state.selected_assistant_title or st.session_state.selected_assistant
        st.subheader(f"📁 Files — {asst_name}")
        hdr1, hdr2, hdr3 = st.columns([5,1,1])
        with hdr2:
            if st.button("Refresh Files", key="refresh_files_btn"):
                st.rerun()
        with hdr3:
            if st.button("Close Files", key="close_files_btn"):
                st.session_state.selected_assistant = None
                st.session_state.selected_assistant_title = None
                st.rerun()

        # List files
        files, ferr = list_assistant_files(current_key, st.session_state.selected_assistant)
        if ferr:
            st.error(ferr)
        elif not files:
            st.info("No files uploaded yet.")
        else:
            for f in files:
                # Extract fields robustly
                if isinstance(f, dict):
                    fid = f.get("id") or f.get("file_id") or f.get("name")
                    fname = f.get("name") or f.get("filename") or fid
                    fstatus = f.get("status") or f.get("state")
                    fsize = f.get("size") or f.get("bytes")
                else:
                    fid = getattr(f, "id", None) or getattr(f, "file_id", None) or getattr(f, "name", None)
                    fname = getattr(f, "name", None) or getattr(f, "filename", None) or fid
                    fstatus = getattr(f, "status", None) or getattr(f, "state", None)
                    fsize = getattr(f, "size", None) or getattr(f, "bytes", None)

                icon = file_status_icon(fstatus or "") if file_status_icon else "📄"
                row = st.columns([6, 2, 2, 1])
                with row[0]:
                    st.write(f"{icon} **{fname}**")
                    st.caption(f"id: `{fid}`")
                with row[1]:
                    st.write(f"Status: {fstatus or 'unknown'}")
                with row[2]:
                    st.write(f"Size: {fsize if fsize is not None else '—'}")
                with row[3]:
                    if delete_assistant_file is not None and st.button("🗑️", key=f"del_file_{fid}"):
                        ok, derr = delete_assistant_file(current_key, st.session_state.selected_assistant, fid)
                        if ok:
                            st.success("Deleted file.")
                        else:
                            st.error(derr or "Failed to delete file.")
                        st.rerun()

        if st.session_state.upload_mode:
            st.markdown("---")
            st.caption("Upload a new file (PDF)")
            with st.form("upload_file_form"):
                up = st.file_uploader("Choose PDF", type=["pdf"], key="upload_pdf")
                meta = st.text_input("Optional metadata (key=value,key2=value2)", key="upload_meta")
                do_upload = st.form_submit_button("Upload")
            if do_upload:
                if not up:
                    st.error("Please select a PDF file.")
                else:
                    # Write to a temporary path
                    tmp_dir = Path("data/tmp")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = tmp_dir / up.name
                    with open(tmp_path, "wb") as fh:
                        fh.write(up.getbuffer())
                    # Parse metadata
                    mdict = None
                    if meta.strip():
                        mdict = {}
                        for pair in meta.split(","):
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                mdict[k.strip()] = v.strip()
                    res, uerr = upload_file_to_assistant(current_key, st.session_state.selected_assistant, str(tmp_path), mdict)
                    if uerr:
                        st.error(uerr)
                    else:
                        st.success(f"Uploaded '{up.name}'.")
                        st.rerun()

if (
    not st.session_state.show_settings
    and current_key
    and st.session_state.chat_open
    and st.session_state.selected_assistant
    and chat_with_assistant is not None
):
    with st.container(border=True):
        asst_name = st.session_state.selected_assistant_title or st.session_state.selected_assistant
        st.subheader(f"💬 Chat — {asst_name}")

        # Header actions
        h1, h2, h3 = st.columns([5,1,1])
        with h2:
            if st.button("Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_history[st.session_state.selected_assistant] = []
                st.rerun()
        with h3:
            if st.button("Close Chat", key="close_chat_btn"):
                st.session_state.chat_open = False
                st.rerun()

        # Show history with clear user/assistant bubbles
        st.markdown(
            """
            <style>
            .stChatMessageContent p { margin-bottom: 0.35rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        hist = st.session_state.chat_history.get(st.session_state.selected_assistant, [])
        if hist:
            for m in hist:
                role = m.get("role", "user")
                msg = m.get("content", "")
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(msg)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg)
        else:
            st.caption("Start a conversation below.")

        # Input
        with st.form("chat_send_form"):
            q = st.text_area("Type your question", height=90, key="chat_input")
            c1, c2, c3, c4 = st.columns([1.2,1,1,1])
            with c1:
                send = st.form_submit_button("Send", type="primary")
            with c2:
                model = st.selectbox("Model", options=["gpt-4o", "gpt-4.1", "o4-mini", "claude-3-5-sonnet", "gemini-2.5-pro"], index=0, key="chat_model")
            with c3:
                temp = st.number_input("Temp", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="chat_temp")
            with c4:
                include_hl = st.checkbox("Highlights", value=False, key="chat_hl")

        if send:
            if not q or not q.strip():
                st.error("Please enter a question.")
            else:
                # append user message
                if st.session_state.selected_assistant not in st.session_state.chat_history:
                    st.session_state.chat_history[st.session_state.selected_assistant] = []
                st.session_state.chat_history[st.session_state.selected_assistant].append({"role": "user", "content": q.strip()})

                # Build messages (last N for context)
                msgs = st.session_state.chat_history[st.session_state.selected_assistant][-10:]

                # Optional context tuning
                context_opts = {"snippet_size": 2000, "top_k": 8}

                resp, cerr = chat_with_assistant(
                    current_key,
                    st.session_state.selected_assistant,
                    msgs,
                    model=model,
                    json_response=False,
                    stream=False,
                    include_highlights=include_hl,
                    context_options=context_opts,
                    temperature=temp,
                )
                if cerr:
                    st.error(cerr)
                else:
                    # Extract assistant text
                    content = ""
                    if isinstance(resp, dict):
                        content = (resp.get("message") or {}).get("content") or ""
                    if not content and isinstance(resp, dict):
                        content = str((resp.get("message") or {}).get("content", ""))

                    st.session_state.chat_history[st.session_state.selected_assistant].append(
                        {"role": "assistant", "content": content or "(no content)"}
                    )

                    # Show citations if present
                    cites = resp.get("citations") if isinstance(resp, dict) else None
                    if cites:
                        with st.expander("Sources & Citations", expanded=False):
                            try:
                                for c in cites:
                                    refs = c.get("references", []) if isinstance(c, dict) else []
                                    for r in refs:
                                        fobj = r.get("file", {}) if isinstance(r, dict) else {}
                                        fname = fobj.get("name") or fobj.get("id") or "document"
                                        pages = r.get("pages") or []
                                        st.markdown(f"- **{fname}** (pages: {', '.join(map(str,pages)) if pages else '—'})")
                            except Exception:
                                st.caption("Citations available but could not be parsed.")

                st.rerun()
