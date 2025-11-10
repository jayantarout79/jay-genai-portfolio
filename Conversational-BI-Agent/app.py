import os
import pandas as pd
import altair as alt
 # ---- Altair theme (consistent palette + higher contrast)
alt.themes.register(
    "bi_theme",
    lambda: {
        "config": {
            "background": "#FFFFFF",
            "axis": {"labelColor": "#334155", "titleColor": "#0f172a"},
            "view": {"stroke": "#E2E8F0"},
            "range": {
                "category": [
                    "#6366f1", "#7c3aed", "#06b6d4", "#22c55e", "#f59e0b",
                    "#ef4444", "#14b8a6", "#a78bfa", "#84cc16", "#f472b6"
                ]
            },
        }
    },
)
alt.themes.enable("bi_theme")

import streamlit as st
from dotenv import load_dotenv
import hashlib

from html import escape
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Utility: handle various return formats from streamlit_mic_recorder
def _extract_audio_bytes(raw):
    if raw is None:
        return None
    # raw may be bytes/bytearray/memoryview OR a dict like {"bytes": b"..."}
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, dict):
        b = raw.get("bytes") or raw.get("audio") or raw.get("data")
        if b is not None:
            try:
                return bytes(b)
            except Exception:
                pass
    return None

def _call_with_timeout(fn, timeout_sec, *args, **kwargs):
    """Run a blocking function in a worker thread and return its result or raise Timeout."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_sec)

# Local services
from services.gemini_client import synthesize_sql
from services.sql_guard import validate_and_fix
from services.snowflake_client import run_query
from services.stt import transcribe_with_openai

# Load local .env for dev (Streamlit Cloud should use Secrets)
load_dotenv()

# Propagate Streamlit Secrets to env (for cloud deploys) if not already set
try:
    if hasattr(st, "secrets"):
        for key in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
        ]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

st.set_page_config(page_title="Conversational BI — Snowflake + Gemini", layout="wide")

# --- Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role: "user"|"assistant", content: str}
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = ""
if "last_chart_spec" not in st.session_state:
    st.session_state.last_chart_spec = None
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "cancel" not in st.session_state:
    st.session_state.cancel = False
if "max_retries" not in st.session_state:
    st.session_state.max_retries = 2  # default retries for model/query
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "rec_key" not in st.session_state:
    st.session_state.rec_key = 0  # used to force-reset the mic component after submit
if "last_audio_sig" not in st.session_state:
    st.session_state.last_audio_sig = ""  # SHA1 of last processed audio to de‑dupe

# --- Enhanced Professional Styling
st.markdown("""
    <style>
        /* ===== Color System (tweak here for quick theme changes) ===== */
        :root{
            --bg:#0b1220;               /* app background */
            --card:#ffffff;             /* cards / containers */
            --text:#0f172a;             /* primary text on light */
            --muted:#475569;            /* secondary text */
            --border:#e2e8f0;           /* card & table borders */
            --brand:#6366f1;            /* indigo */
            --brand-2:#7c3aed;          /* violet */
            --brand-10:#6366f11a;       /* brand translucent */
            --success:#10b981;
            --danger:#ef4444;
        }

        /* Main container */
        .main {
            background: radial-gradient(1200px 600px at 20% -10%, #1f293766 0%, transparent 60%),
                        radial-gradient(900px 500px at 100% 10%, #312e8180 0%, transparent 60%),
                        var(--bg);
            padding: 0;
        }

        /* Header */
        .header-container {
            background: var(--card);
            color: var(--text);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin: 1.5rem 0 2rem 0;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
            border: 1px solid var(--border);
        }

        .header-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        /* Connection badge */
        .connection-status {
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-size: 0.92rem;
            color: #1f2937;
            border: 1px solid #c7d2fe;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 0 3px #10b98133;
        }

        /* Chat container */
        .chat-container {
            background: var(--card);
            color: var(--text);
            border-radius: 16px;
            padding: 1.5rem 1.5rem 0.5rem;
            margin: 0 0 2rem 0;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
            border: 1px solid var(--border);
            min-height: 0; /* min-height removed to prevent large empty box when there are no messages */
        }

        /* Chat messages */
        [data-testid="stChatMessageContent"] {
            background: #f8fafc;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            border: 1px solid var(--border);
            color: var(--text);
        }
        /* User bubble: strong contrast */
        [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%);
            color: #ffffff;
            border: none;
        }
        /* Assistant bubble */
        [data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageContent"] {
            background: #ffffff;
            border: 1px solid var(--border);
        }

        /* Chat input – improved readability & focus ring */
        [data-testid="stChatInput"] {
            position: relative !important;
            border-radius: 24px !important;
            border: 2px solid #c7d2fe !important;
            background: #111827 !important;
            box-shadow: inset 0 0 0 1px #00000033, 0 6px 16px rgba(99,102,241,.25) !important;
        }
        [data-testid="stChatInput"] input, 
        [data-testid="stChatInput"] textarea {
            color: #E5E7EB !important;
            caret-color: var(--brand) !important;
        }
        [data-testid="stChatInput"] ::placeholder {
            color: #9CA3AF !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--brand) !important;
            box-shadow: 0 0 0 3px var(--brand-10) !important;
        }

        /* Mic button: pinned inside chat input (right side) */
        iframe[title="streamlit_mic_recorder.st_mic_recorder"] {
            position: absolute !important;
            bottom: 10px !important;
            right: 64px !important;
            width: 40px !important;
            height: 40px !important;
            z-index: 5 !important;
            border-radius: 50% !important;
            filter: drop-shadow(0 3px 10px rgba(0,0,0,.25));
        }

        /* Code blocks */
        code {
            background: #f1f5f9 !important;
            border-radius: 6px !important;
            padding: 0.2rem 0.4rem !important;
            color: var(--brand) !important;
        }
        pre {
            background: #0b1220 !important;
            color: #e5e7eb !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: 1px solid #1f2937 !important;
        }

        /* Buttons */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.25rem !important;
            border: none !important;
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%) !important;
            color: #fff !important;
        }
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 18px rgba(99,102,241,.35) !important;
        }
        button[key="stop_btn"]{ background: var(--danger) !important; }
        button[key="stop_btn"]:hover{ background:#dc2626 !important; }

        /* Copy SQL button */
        .copy-sql-btn {
            background: linear-gradient(135deg, var(--brand) 0%, var(--brand-2) 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: transform .15s ease, box-shadow .15s ease !important;
            font-size: 0.9rem !important;
        }
        .copy-sql-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(99,102,241,.30) !important;
        }

        /* Tables / dataframes */
        [data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
        }

        /* Alerts */
        .stAlert {
            border-radius: 12px !important;
            border-left: 4px solid var(--brand) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0f172a !important;
            color: #E5E7EB !important;
            border-right: 1px solid #111827 !important;
        }
        [data-testid="stSidebar"] h2 {
            color: #c7d2fe !important;
            font-weight: 700 !important;
        }

        /* Divider */
        hr {
            margin: 2rem 0 !important;
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, #c7d2fe, transparent) !important;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Mobile */
        @media (max-width: 768px) {
            .header-title { font-size: 1.9rem; }
            .chat-container { padding: 1rem; }
            iframe[title="streamlit_mic_recorder.st_mic_recorder"] {
                bottom: 8px !important;
                right: 56px !important;
                width: 36px !important;
                height: 36px !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section
st.markdown("""
    <div class="header-container">
        <div class="header-title">
            🎤 Conversational BI
        </div>
        <p style="color: #64748b; font-size: 1.1rem; margin: 0.5rem 0 1rem 0;">
            Ask questions in natural language and get instant insights from your Snowflake data
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar: Env sanity
with st.sidebar:
    st.header("⚙️ Configuration")
    missing = []
    for k in [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]:
        if not os.environ.get(k):
            missing.append(k)
    if missing:
        st.warning("⚠️ Missing: " + ", ".join(missing))
    else:
        st.success("✅ All credentials configured")
    
    st.caption("💡 Configure via .env (local) or Streamlit Secrets (cloud)")
    
    st.divider()

    # --- Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sql = ""
        st.session_state.last_explanation = ""
        st.session_state.last_chart_spec = None
        st.session_state.awaiting_response = False
        st.session_state.stop_requested = False
        st.session_state.last_df = None
        st.rerun()

# --- Connection Health Check (stop early if broken)
try:
    df_check, _ = run_query("select current_account() as acct, current_region() as region")
    # Normalize column names to lowercase to avoid case-sensitivity issues
    df_check.columns = [str(c).lower() for c in df_check.columns]
    if not df_check.empty:
        st.markdown(f"""
            <div class="connection-status">
                <span class="status-dot"></span>
                <span>Connected to Snowflake: <strong>{df_check.iloc[0]['acct']}</strong> • {df_check.iloc[0]['region']}</span>
            </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Snowflake connection failed: {e}")
    st.stop()

st.divider()

# --- Chat UI with professional container
if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    # Render chat history only when messages exist
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    st.markdown('</div>', unsafe_allow_html=True)
# If there are no messages, don't render the white chat container at all.

# Text chat input as fallback
user_text = st.chat_input("Ask your data anything... 💬")

# --- Mic recorder integrated into the chat input
wav_bytes = None

if os.environ.get("OPENAI_API_KEY"):
    try:
        from streamlit_mic_recorder import mic_recorder

        if not st.session_state.awaiting_response:
            raw_audio = mic_recorder(
                start_prompt="🎙️",
                stop_prompt="⏹",
                just_once=True,
                format="wav",
                key=f"rec_chat_{st.session_state.rec_key}",
                use_container_width=False,
            )
            wav_bytes = _extract_audio_bytes(raw_audio)
        else:
            wav_bytes = None
    except Exception:
        wav_bytes = None
else:
    st.caption("💡 Set OPENAI_API_KEY to enable voice transcription")

# If voice captured, transcribe and enqueue as user message, de-duping same audio
if wav_bytes and not st.session_state.awaiting_response:
    sig = hashlib.sha1(wav_bytes).hexdigest()
    if sig != st.session_state.last_audio_sig:
        try:
            transcript = transcribe_with_openai(wav_bytes)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                st.session_state.last_audio_sig = sig
                st.session_state.awaiting_response = True
                st.session_state.rec_key += 1
                st.rerun()
        except Exception as e:
            st.warning(f"⚠️ Transcription failed: {e}")

if user_text and not st.session_state.awaiting_response:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.last_audio_sig = ""  # reset audio de-dupe on manual text
    st.session_state.awaiting_response = True
    st.rerun()

# If last message is user and we are awaiting response, process NL→SQL→Snowflake
def _need_model_response():
    msgs = st.session_state.messages
    return bool(msgs) and msgs[-1]["role"] == "user" and st.session_state.awaiting_response

if _need_model_response():
    q = st.session_state.messages[-1]["content"]

    # Controls and live status (persistent Stop)
    ctrl_cols = st.columns([1, 0.12])
    status_ph = ctrl_cols[0].empty()
    if ctrl_cols[1].button("⏹ Stop", key="stop_btn", help="Cancel this run"):
        st.session_state.stop_requested = True
        st.session_state.cancel = True
        st.session_state.awaiting_response = False
        status_ph.warning("Cancelling…")
        st.stop()

    # If a new run begins with stop requested, immediately exit
    if st.session_state.stop_requested:
        st.session_state.awaiting_response = False
        st.session_state.cancel = True
        status_ph.warning("Cancelled by user.")
        st.session_state.stop_requested = False  # clear for next request
        st.stop()

    st.session_state.cancel = False

    try:
        last_err = None
        for attempt in range(1, int(st.session_state.max_retries) + 1):
            if st.session_state.stop_requested or st.session_state.cancel:
                raise RuntimeError("Cancelled by user")

            status_ph.info(f"🔄 Processing your request... (attempt {attempt}/{st.session_state.max_retries})")

            # 1) NL -> SQL (Gemini)
            try:
                if st.session_state.stop_requested or st.session_state.cancel:
                    raise RuntimeError("Cancelled by user")
                try:
                    model = _call_with_timeout(synthesize_sql, 20, q)  # 20s timeout for model call
                except FuturesTimeout:
                    last_err = TimeoutError("Model timed out")
                    if attempt < st.session_state.max_retries:
                        continue
                    raise last_err
                if st.session_state.stop_requested or st.session_state.cancel:
                    raise RuntimeError("Cancelled by user")
                st.session_state.last_chart_spec = getattr(model, "chart", None)
                st.session_state.last_explanation = getattr(model, "explanation", "")
                safe_sql = validate_and_fix(model.sql)
                st.session_state.last_sql = safe_sql
            except Exception as e:
                last_err = e
                if attempt < st.session_state.max_retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise

            # 2) Execute SQL (Snowflake)
            try:
                if st.session_state.stop_requested or st.session_state.cancel:
                    raise RuntimeError("Cancelled by user")
                try:
                    df, meta = _call_with_timeout(run_query, 20, st.session_state.last_sql)  # 20s timeout for DB call
                except FuturesTimeout:
                    last_err = TimeoutError("Query timed out")
                    if attempt < st.session_state.max_retries:
                        continue
                    raise last_err
                st.session_state.last_df = df.copy()
            except Exception as e:
                last_err = e
                if attempt < st.session_state.max_retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise

            # Success -> compose assistant reply and break
            summary_lines = [f"**📊 Results:** {len(df)} rows"]
            if isinstance(meta, dict) and "rowcount" in meta:
                summary_lines.append(f"Rowcount: {meta['rowcount']}")
            summary_lines.append("")
            summary_lines.append("```sql")
            summary_lines.append(st.session_state.last_sql)
            summary_lines.append("```")
            st.session_state.messages.append({"role": "assistant", "content": "\n".join(summary_lines)})
            break

    except Exception as e:
        err = last_err or e
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {err}"})
    finally:
        if st.session_state.cancel:
            status_ph.warning("⏹ Cancelled.")
        status_ph.empty()
        st.session_state.awaiting_response = False
        # do not rerun if we stopped; otherwise rerun to show results
        if not st.session_state.cancel:
            st.rerun()
        else:
            st.session_state.cancel = False

# Render rich payload for the latest assistant message (SQL + copy + table + chart + explanation)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.last_sql:
    with st.chat_message("assistant"):
        st.markdown("### 📝 Generated SQL Query")
        st.code(st.session_state.last_sql, language="sql")

        # Copy SQL button with professional styling
        escaped_sql = escape(st.session_state.last_sql)
        st.markdown(
            f"""
            <div style="margin: 1rem 0;">
              <button onclick="navigator.clipboard.writeText(document.getElementById('sql-to-copy').textContent)" 
                      class="copy-sql-btn">
                  📋 Copy SQL to Clipboard
              </button>
            </div>
            <pre id="sql-to-copy" style="position:absolute; left:-9999px; top:-9999px;">{escaped_sql}</pre>
            """,
            unsafe_allow_html=True,
        )

        # Use cached df from last run if present, else execute once more
        try:
            df = st.session_state.last_df
            if df is None:
                df, _ = run_query(st.session_state.last_sql)
            if df is not None and not df.empty:
                st.markdown("### 📊 Query Results")
                st.dataframe(df, use_container_width=True)

                # Build chart from stored spec with fallbacks
                chart_spec = st.session_state.last_chart_spec
                try:
                    x = getattr(chart_spec, "x", None) if chart_spec else None
                    y = getattr(chart_spec, "y", None) if chart_spec else None
                    agg = getattr(chart_spec, "aggregate", None) if chart_spec else None
                    kind = getattr(chart_spec, "type", None) if chart_spec else None
                except Exception:
                    x = y = agg = kind = None

                if (not x) or (x not in df.columns):
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    x = cat_cols[0] if cat_cols else None
                if (not y) or (y not in df.columns):
                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    y = num_cols[0] if num_cols else None
                    agg = None

                if x and y and x in df.columns and y in df.columns:
                    st.markdown("### 📈 Data Visualization")
                    kind = kind or "bar"
                    if kind in ("bar", "line", "area"):
                        mark = {"bar": "bar", "line": "line", "area": "area"}[kind]
                        chart = getattr(alt.Chart(df), f"mark_{mark}")()
                        y_enc = alt.Y(f"{agg}({y}):Q", sort=alt.EncodingSortField(field=y, order="descending")) if agg else alt.Y(y, sort=alt.EncodingSortField(field=y, order="descending"))
                        chart = chart.encode(x=alt.X(x, sort=None), y=y_enc).properties(height=420)
                        st.altair_chart(chart, use_container_width=True)
                    elif kind == "pie":
                        theta = f"{agg}({y})" if agg else y
                        chart = alt.Chart(df).mark_arc().encode(theta=alt.Theta(theta), color=x).properties(height=420)
                        st.altair_chart(chart, use_container_width=True)
            else:
                st.info("ℹ️ No data returned from query.")

            if st.session_state.last_explanation:
                st.markdown("### 💡 Explanation")
                st.write(st.session_state.last_explanation)
        except Exception as e:
            st.warning(f"⚠️ Render failed: {e}")

st.markdown("---")
st.caption("💡 **Tip:** Use the ⏹ Stop button to cancel requests. Each step has a 20s timeout. Retries: " + str(st.session_state.max_retries))