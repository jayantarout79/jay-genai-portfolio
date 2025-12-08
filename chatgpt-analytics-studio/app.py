import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import base64
import io
import altair as alt
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types as genai_types
from PIL import Image
from openai import OpenAI

from services.ai_analysis import (
    run_openai_analysis,
    run_parallel_analysis,
    save_ai_aggregates,
)
from services.early_insights import compute_early_insights
from services.fingerprint import compute_folder_fingerprint
from services.preprocessing import chunk_conversations, generate_audio_transcripts
from services.state_manager import (
    STATE_PATH,
    ensure_state_basics,
    fingerprint_matches,
    load_state,
    save_state,
)

RAW_FOLDER = "chatgpt_raw_data"
CHUNKS_DIR = os.path.join("data", "chunks")
TRANSCRIPTS_DIR = os.path.join("data", "audio_transcripts")
AGGREGATES_PATH = os.path.join("data", "ai_aggregates.json")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

st.set_page_config(page_title="ChatGPT Analytics Studio", layout="wide")

# Lightweight .env loader to make sure OPENAI_API_KEY is available even if Streamlit starts from another cwd.
def _load_env():
    candidates = []
    cwd_env = os.path.join(os.getcwd(), ".env")
    candidates.append(cwd_env)
    here_env = os.path.join(os.path.dirname(__file__), ".env")
    candidates.append(here_env)
    # walk up parents
    cur = os.path.abspath(__file__)
    for _ in range(5):
        cur = os.path.dirname(cur)
        candidates.append(os.path.join(cur, ".env"))
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        if "=" in line and not line.strip().startswith("#"):
                            k, v = line.strip().split("=", 1)
                            os.environ.setdefault(k, v)
                break
            except OSError:
                continue

_load_env()


@st.cache_data(show_spinner=False)
def get_raw_fingerprint(path: str) -> Any:
    # Cached to avoid recomputing if user navigates within the session.
    return compute_folder_fingerprint(path)


def load_ai_aggregates() -> Dict[str, Any]:
    if not os.path.isfile(AGGREGATES_PATH):
        return {}
    try:
        with open(AGGREGATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def render_step_header(label: str):
    st.markdown(f"### {label}")


def clear_cached_outputs():
    """Drop processing outputs so user can re-run from scratch without touching raw data."""
    for path in [STATE_PATH, AGGREGATES_PATH]:
        if os.path.isfile(path):
            os.remove(path)
    for dir_path in [CHUNKS_DIR, TRANSCRIPTS_DIR]:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def build_profile_prompt(final_insights: Dict[str, Any], top_topic: str) -> str:
    summary = final_insights.get("overall_summary") or "A professional ChatGPT user focused on practical outcomes."
    profile = final_insights.get("user_profile") or "Engaged, curious user."
    tone = final_insights.get("tone_style") or "Friendly and focused."
    prompting = final_insights.get("prompting_level") or "Intermediate"
    return (
        "Create a polished square portrait-style illustration of a ChatGPT power user. "
        "Style: clean, modern, product-like card; soft gradients; minimal background; friendly but professional. "
        f"Theme: {top_topic}. "
        f"Persona summary: {summary}. "
        f"User profile: {profile}. "
        f"Tone/style: {tone}. "
        f"Prompting level: {prompting}. "
        "Include subtle UI motifs or icons that imply AI and productivity. No text in the image."
    )


def build_dashboard_prompt(user_handle: str, metrics: Dict[str, Any], pq: Dict[str, float], top_topic: str) -> str:
    return f"""
Create a clean, modern data dashboard graphic (16:9, ultra-high resolution, LinkedIn-style) summarizing a user's AI Prompt Persona.

Content (use these exact values):
- Title: “Prompt Persona Summary — {user_handle}”
- Highlight badges:
  • Prompting Level: {metrics.get('prompting_level','Intermediate')}
  • Tone: {metrics.get('tone','Analytical & Constructive')}
  • Top Topic: {top_topic or 'Business / Productivity'}
- Description box:
  “{metrics.get('description','Strong focus on business productivity, software engineering, and creative content creation. Mix of intermediate to advanced prompting. Sentiment indicates a need for technical guidance + creative strategy.')}”
- KPI cards:
  • Messages: {metrics.get('messages', '0')}
  • Distinct Words: {metrics.get('distinct_words', '0')}
  • Audio Minutes: {metrics.get('audio_minutes', '0')}
  • Prompt Quality: Beginner {pq.get('Beginner',0)}% · Intermediate {pq.get('Intermediate',0)}% · Advanced {pq.get('Advanced',0)}%

Design style:
- Apple-style UI, soft neumorphism, rounded cards, soft gradients.
- Colors: navy blue, teal, soft white, subtle shadows.
- Typography: clean, minimal, professional.
- Include subtle icons for messages, audio, words, and quality.
- Premium analytics dashboard look for creators.
- 16:9 aspect ratio, ultra high-resolution, crisp, no pixelation.
- No extra text beyond the specified content.
"""


def generate_profile_image(prompt: str) -> Dict[str, Any]:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=genai_types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="2K",
                    ),
                ),
            )
            for part in resp.parts:
                try:
                    img = part.as_image()
                    if img:
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        return {"image_bytes": buf.getvalue()}
                except Exception:
                    pass
                inline = getattr(part, "inline_data", None)
                if inline is not None:
                    raw = getattr(inline, "data", None)
                    if raw is None:
                        continue
                    try:
                        img = Image.open(io.BytesIO(raw))
                        buf = io.BytesIO()
                        img.save(buf, "PNG")
                        return {"image_bytes": buf.getvalue()}
                    except Exception:
                        return {"image_bytes": raw}
            return {"error": "No image data returned from Gemini."}
        except Exception as e:
            return {"error": f"Gemini error: {e}"}
    return {"error": "No image provider available (set GEMINI_API_KEY)."}


def _as_bullets(val: Any) -> str:
    if isinstance(val, list):
        return "<br>".join([f"• {str(v)}" for v in val])
    if isinstance(val, str):
        return val
    return ""


def monthly_sentiment_from_messages(raw_folder: str) -> List[Dict[str, Any]]:
    from services import data_loader
    messages = data_loader.flatten_messages(data_loader.load_conversations(raw_folder), author_filter="user")
    if not messages:
        return []
    pos_words = {"good", "great", "love", "happy", "nice", "win", "cool", "thanks", "awesome", "excellent"}
    neg_words = {"bad", "sad", "angry", "hate", "issue", "problem", "bug", "fail", "error", "frustrated"}
    buckets: Dict[str, List[float]] = {}
    samples: Dict[str, Dict[str, str]] = {}
    for m in messages:
        text = (m.get("text") or "").lower()
        created_at = m.get("created_at")
        if not created_at:
            continue
        dt = datetime.fromtimestamp(created_at)
        key = f"{dt.year:04d}-{dt.month:02d}"
        pos = sum(w in pos_words for w in text.split())
        neg = sum(w in neg_words for w in text.split())
        score = 0
        if pos + neg > 0:
            score = (pos - neg) / (pos + neg)
        buckets.setdefault(key, []).append(score)
        if score > 0.5 and "high" not in samples:
            samples["high"] = {"month": key, "text": m.get("text", "")}
        if score < -0.2:
            samples["low"] = {"month": key, "text": m.get("text", "")}
    out = []
    for k, vals in buckets.items():
        avg = sum(vals) / len(vals)
        out.append({"month": k, "sentiment": round(avg, 3)})
    out = sorted(out, key=lambda x: x["month"])
    return out, samples


def main():
    # Top-level UI and pipeline orchestration: run pipeline and render dashboards.
    st.title("ChatGPT Analytics Studio")
    st.caption("Process your export and surface insights with one click.")

    raw_exists = os.path.isdir(RAW_FOLDER)
    raw_fingerprint = get_raw_fingerprint(RAW_FOLDER) if raw_exists else None
    state = ensure_state_basics(RAW_FOLDER, raw_fingerprint) if raw_exists else load_state()
    buttons_disabled = not raw_exists

    if not raw_exists:
        st.error("Raw data folder missing. Please place it at project root under chatgpt_raw_data/")
        st.stop()

    col_run, col_ai, col_clear = st.columns([1, 1, 1])
    env_has_key = bool(os.getenv("OPENAI_API_KEY"))
    use_openai_default = env_has_key
    with col_run:
        run_full = st.button("▶️ Run from start", type="primary", disabled=buttons_disabled)
    with col_ai:
        run_ai_only = st.button("🤖 Run AI analysis only", disabled=buttons_disabled)
    with col_clear:
        if st.button("🧹 Clear cache"):
            clear_cached_outputs()
            st.success("Cache cleared. Reloading app.")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

    def do_preprocessing():
        # Chunk user-only prompts and build transcript placeholders.
        with st.spinner("Preprocessing: chunking user prompts and scanning audio..."):
            num_chunks, chunk_paths = chunk_conversations(RAW_FOLDER, CHUNKS_DIR)
            num_transcripts, transcript_paths = generate_audio_transcripts(RAW_FOLDER, TRANSCRIPTS_DIR)
            pre_data = {
                "chunks_dir": CHUNKS_DIR,
                "chunk_paths": chunk_paths,
                "num_chunks": num_chunks,
                "audio_transcripts_dir": TRANSCRIPTS_DIR,
                "transcript_paths": transcript_paths,
                "num_audio_transcripts": num_transcripts,
                "processed_at": datetime.utcnow().isoformat() + "Z",
            }
            state["preprocessing"] = pre_data
            save_state(state)
            return pre_data

    def do_ai_analysis(pre_data, use_openai: bool):
        # Run heuristic or OpenAI analysis on chunks/transcripts.
        with st.spinner("Running AI analysis..."):
            chunk_paths = pre_data.get("chunk_paths") or []
            transcript_paths = pre_data.get("transcript_paths") or []
            if use_openai:
                aggregates = run_openai_analysis(chunk_paths, transcript_paths)
            else:
                aggregates = run_parallel_analysis(chunk_paths, transcript_paths)
            save_ai_aggregates(aggregates, AGGREGATES_PATH)
            ai_state = {
                "aggregates_path": AGGREGATES_PATH,
                "models_used": aggregates.get("models_used", []),
                "analysis_at": datetime.utcnow().isoformat() + "Z",
            }
            state["ai_analysis"] = ai_state
            save_state(state)
            return aggregates

    aggregates = load_ai_aggregates()
    early_data: Dict[str, Any] = state.get("early_insights", {})
    pre_data: Dict[str, Any] = state.get("preprocessing", {})

    if run_full:
        with st.spinner("Running full pipeline..."):
            early_data = compute_early_insights(RAW_FOLDER)
            state["early_insights"] = early_data
            save_state(state)
            pre_data = do_preprocessing()
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("OPENAI_API_KEY not set; falling back to heuristic analysis.")
            use_openai = use_openai_default and bool(os.getenv("OPENAI_API_KEY"))
            aggregates = do_ai_analysis(pre_data, use_openai)
        st.success("Pipeline complete.")

    if run_ai_only:
        if not pre_data:
            pre_data = do_preprocessing()
        if not early_data:
            early_data = compute_early_insights(RAW_FOLDER)
            state["early_insights"] = early_data
            save_state(state)
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY not set; falling back to heuristic analysis.")
        use_openai = use_openai_default and bool(os.getenv("OPENAI_API_KEY"))
        aggregates = do_ai_analysis(pre_data, use_openai)
        st.success("AI analysis complete.")

    if not (early_data and aggregates):
        st.info("Run from start or Run AI analysis to view the report.")
        return

    user = early_data.get("user", {})
    messages = early_data.get("messages", {})
    audio = early_data.get("audio", {})
    images = early_data.get("images", {})
    top_topic = (aggregates.get("top_topics") or [{"topic": "technology"}])[0].get("topic", "technology")

    # --- Theming to mimic the provided design ---
    st.markdown(
        """
        <style>
        .main, .block-container {background: radial-gradient(circle at 20% 20%, rgba(22,38,70,0.8), rgba(6,10,20,0.95)); color: #e8f3ff;}
        .section-card {background: rgba(20,30,50,0.7); border: 1px solid rgba(78,255,255,0.25); border-radius: 16px; padding: 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.35);}
        .kpi-row {display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 12px 0;}
        .kpi-card {background: linear-gradient(145deg, rgba(40,60,90,0.9), rgba(18,26,46,0.9)); border-radius: 14px; padding: 12px 14px; border: 1px solid rgba(93,243,255,0.4); box-shadow: inset 0 1px 8px rgba(255,255,255,0.08), 0 10px 25px rgba(0,0,0,0.35); display:flex; align-items:center; gap:10px;}
        .kpi-title {font-size: 11px; color: #a8c7ff; margin-bottom: 4px;}
        .kpi-value {font-size: 20px; font-weight: 700; color: #8cf0ff;}
        .kpi-icon {width:28px; height:28px; border-radius:10px; background:linear-gradient(135deg,#6bf0ff,#7f8cff); display:flex; align-items:center; justify-content:center; font-size:14px; color:#0b1220; flex-shrink:0;}
        .badge-row {display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px;}
        .badge {padding: 6px 12px; border-radius: 999px; font-weight: 600; color: #0f172a;}
        .badge-sub {background: linear-gradient(135deg, #e9c27d, #d89a34); color: #0e0e0e;}
        .panel {margin-top: 14px;}
        .box-heading {color:#a8c7ff; font-weight:700; margin-bottom:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab_labels = ["Early Insights", "AI Insights", "AI Final Insights", "Nano Bana Profile"]
    tab_early, tab_ai, tab_final, tab_card, tab_sentiment, tab_chat = st.tabs(
        ["Early Insights", "AI Insights", "AI Final Insights", "Nano Bana Profile", "Sentiment Curve", "Chat"]
    )

    def prompt_quality_distribution(items: List[Dict]) -> Dict[str, float]:
        counts = {"Beginner": 0, "Intermediate": 0, "Advanced": 0, "Power": 0}
        total = 0
        for it in items:
            label = it.get("prompt_quality_label")
            if label in counts:
                counts[label] += 1
                total += 1
            for pm in it.get("per_message", []) or []:
                lbl = pm.get("prompt_quality_label")
                if lbl in counts:
                    counts[lbl] += 1
                    total += 1
        if total == 0:
            return counts
        return {k: round(v * 100 / total, 1) for k, v in counts.items()}

    with tab_early:
        st.markdown(
            f"""
            <div class="badge-row">
                <span style="color:#e8f3ff;">{user.get('email','N/A')}</span>
                <span style="color:#a8c7ff;">Birth year: {user.get('dob','N/A')}</span>
                <span class="badge badge-sub">{user.get('subscription','ChatGPT Plus')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
        kpis = [
            ("📩", "Total messages", f"{messages.get('total_messages', 0):,}"),
            ("✏️", "Approx. characters", f"{messages.get('approx_chars', 0):,}"),
            ("🔤", "Distinct words used", f"{messages.get('distinct_words', 0):,}"),
            ("🎧", "Audio conversations minutes", f"{audio.get('minutes', 0.0):,.1f}"),
            ("📤", "Images uploaded", f"{images.get('uploaded', 0)}"),
            ("✨", "Images generated", f"{images.get('generated', 0)}"),
        ]
        for icon, title, value in kpis:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-icon">{icon}</div><div><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        monthly = messages.get("by_month", [])
        top_words = messages.get("top_words", [])
        longest_words = messages.get("longest_words", [])

        col_left, col_right = st.columns([2, 1])
        with col_left:
            if monthly:
                df_month = pd.DataFrame(monthly).sort_values("month")
                st.markdown('<div class="section-card panel"><div class="box-heading">Message per month (Time-wise)</div>', unsafe_allow_html=True)
                line = (
                    alt.Chart(df_month)
                    .mark_line(color="#55f1ff", point=True)
                    .encode(
                        x=alt.X("month:N", title=""),
                        y=alt.Y("count:Q", title="", axis=alt.Axis(format="~s")),
                        tooltip=["month", "count"],
                    )
                )
                text = line.mark_text(align="left", dx=4, dy=-6, color="#e8f3ff").encode(text="count:Q")
                st.altair_chart(line + text, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with col_right:
            if top_words:
                st.markdown('<div class="section-card panel"><div class="box-heading">Top 10 words you used</div>', unsafe_allow_html=True)
                dfw = pd.DataFrame(top_words)
                bar = alt.Chart(dfw).mark_bar(color="#55f1ff").encode(
                    y=alt.Y("word:N", sort="-x", title=""),
                    x=alt.X("count:Q", title="", axis=alt.Axis(format="~s")),
                ).properties(height=170)
                text = bar.mark_text(dx=4, dy=0, align="left", baseline="middle", color="#e8f3ff").encode(text="count:Q")
                st.altair_chart(bar + text, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            if longest_words:
                st.markdown('<div class="section-card panel"><div class="box-heading">Top 10 lengthiest words</div>', unsafe_allow_html=True)
                dfl = pd.DataFrame(longest_words)
                bar_l = alt.Chart(dfl).mark_bar(color="#9c7bff").encode(
                    y=alt.Y("word:N", sort="-x", title=""),
                    x=alt.X("count:Q", title="", axis=alt.Axis(format="~s")),
                ).properties(height=170)
                text_l = bar_l.mark_text(dx=4, dy=0, align="left", baseline="middle", color="#e8f3ff").encode(text="count:Q")
                st.altair_chart(bar_l + text_l, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab_ai:
        topics = aggregates.get("top_topics", [])
        if topics:
            df_topics = pd.DataFrame(topics)
            st.markdown('<div class="section-card panel"><div class="box-heading">Top 10 Topics</div>', unsafe_allow_html=True)
            chart_topics = (
                alt.Chart(df_topics)
                .mark_bar(color="#55f1ff")
                .encode(y=alt.Y("topic:N", sort="-x", title="Topic"), x=alt.X("count:Q", title="Count"))
                .properties(height=360)
            )
            text_topics = chart_topics.mark_text(dx=4, dy=0, align="left", baseline="middle", color="#e8f3ff").encode(text="count:Q")
            st.altair_chart(chart_topics + text_topics, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        pq = prompt_quality_distribution(aggregates.get("items", []))
        if pq:
            st.markdown('<div class="section-card panel"><div class="box-heading">Prompt Quality %</div>', unsafe_allow_html=True)
            df_pq = pd.DataFrame([{"level": k, "percent": v} for k, v in pq.items()])
            chart_pq = (
                alt.Chart(df_pq)
                .mark_bar()
                .encode(
                    x=alt.X("percent:Q", title="", axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("level:N", sort="-x", title=""),
                    color=alt.Color("level:N", scale=alt.Scale(range=["#55f1ff", "#7f8cff", "#b57cff", "#5ef7d7"]), legend=None),
                    tooltip=["level", "percent"],
                )
                .properties(height=180)
            )
            text_pq = chart_pq.mark_text(dx=4, dy=0, align="left", baseline="middle", color="#e8f3ff").encode(text=alt.Text("percent:Q", format=".1f"))
            st.altair_chart(chart_pq + text_pq, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_final:
        final_insights = aggregates.get("final_insights", {}) or {}
        st.markdown('<div class="section-card panel">', unsafe_allow_html=True)
        st.markdown("#### AI Final Insights Summary", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:14px;color:#e8f3ff;'>{_as_bullets(final_insights.get('overall_summary',''))}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-card" style="margin-top:10px;">
              <div class="box-heading">Highlights ⭐</div>
              <div>{_as_bullets(final_insights.get('highlights',''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="section-card" style="margin-top:10px;">
              <div class="box-heading">Risks ⚠️</div>
              <div>{_as_bullets(final_insights.get('risks',''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="section-card" style="margin-top:10px;">
              <div class="box-heading">Recommendations 💡</div>
              <div>{_as_bullets(final_insights.get('recommendations',''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="section-card" style="margin-top:10px;">
              <div class="box-heading">Improvement Tips 🚀</div>
              <div>{_as_bullets(final_insights.get('improvement_tips',''))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_sentiment:
        st.subheader("Sentiment Curve")
        sent_data, samples = monthly_sentiment_from_messages(RAW_FOLDER)
        if sent_data:
            df_sent = pd.DataFrame(sent_data)
            line = (
                alt.Chart(df_sent)
                .mark_line(color="#55f1ff", point=True)
                .encode(
                    x=alt.X("month:N", title=""),
                    y=alt.Y("sentiment:Q", title="Sentiment", scale=alt.Scale(domain=[-1, 1])),
                    tooltip=["month", "sentiment"],
                )
            )
            text = line.mark_text(align="left", dx=4, dy=-6, color="#e8f3ff").encode(text=alt.Text("sentiment:Q", format=".2f"))
            st.altair_chart(line + text, use_container_width=True)
            high = samples.get("high")
            low = samples.get("low")
            if high:
                st.success(f"Highest sentiment ({high['month']}): {high.get('text','')}")
            if low:
                st.error(f"Lowest sentiment ({low['month']}): {low.get('text','')}")
        else:
            st.info("No sentiment data available.")

    with tab_card:
        pq = prompt_quality_distribution(aggregates.get("items", []))
        st.markdown("#### ChatGPT Persona Summary")
        dash_metrics = {
            "prompting_level": final_insights.get("prompting_level", "Intermediate"),
            "tone": final_insights.get("tone_style", "Analytical & Constructive"),
            "description": final_insights.get(
                "overall_summary",
                "Strong focus on business productivity, software engineering, and creative content creation. Mix of intermediate to advanced prompting. Sentiment indicates a need for technical guidance + creative strategy.",
            ),
            "messages": f"{messages.get('total_messages', 0):,}",
            "distinct_words": f"{messages.get('distinct_words', 0):,}",
            "audio_minutes": f"{audio.get('minutes', 0.0):,.1f}",
        }
        dash_prompt = build_dashboard_prompt("jayantarout79", dash_metrics, pq, top_topic)
        regen_dash = st.button("Generate dashboard image")
        if regen_dash or "dashboard_image_bytes" not in st.session_state:
            dash_result = generate_profile_image(dash_prompt)
            if dash_result.get("image_bytes"):
                st.session_state["dashboard_image_bytes"] = dash_result["image_bytes"]
                st.session_state.pop("dashboard_image_error", None)
            else:
                st.session_state["dashboard_image_error"] = dash_result.get("error", "Unknown error")
        if st.session_state.get("dashboard_image_error"):
            st.error(f"Dashboard image failed: {st.session_state['dashboard_image_error']}")
        if st.session_state.get("dashboard_image_bytes"):
            st.image(st.session_state["dashboard_image_bytes"], use_column_width=True)
            st.download_button(
                "Download dashboard image",
                data=st.session_state["dashboard_image_bytes"],
                file_name="prompt_persona_dashboard.png",
                mime="image/png",
            )

    with tab_chat:
        st.subheader("Chat over your indexed chunks (Gemini File Search)")
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            st.info("Set GEMINI_API_KEY to enable chat and indexing.")
        client = genai.Client(api_key=api_key) if api_key else None

        @st.cache_resource(show_spinner=False)
        def get_store(_client: genai.Client):
            store = _client.file_search_stores.create()
            return store.name

        store_name = get_store(client) if client else None
        if store_name:
            st.caption(f"Using File Search Store: `{store_name}`")

        if "chunks_indexed" not in st.session_state:
            st.session_state["chunks_indexed"] = False
            st.session_state["indexed_fingerprint"] = None

        chunks = state.get("preprocessing", {}).get("chunk_paths", [])
        already_indexed = (
            st.session_state.get("chunks_indexed")
            and st.session_state.get("indexed_fingerprint") == raw_fingerprint
        )

        index_disabled = not api_key or not chunks
        if st.button("Index chunk files to store", disabled=index_disabled):
            if not api_key:
                st.error("Set GEMINI_API_KEY first.")
            elif not chunks:
                st.error("Run preprocessing so chunk files exist.")
            else:
                try:
                    for path in chunks:
                        client.file_search_stores.upload_to_file_search_store(
                            file_search_store_name=store_name, file=path
                        )
                    st.session_state["chunks_indexed"] = True
                    st.session_state["indexed_fingerprint"] = raw_fingerprint
                    st.success("Chunks uploaded (cached for this fingerprint).")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
        elif already_indexed:
            st.caption("Chunks already indexed for this dataset. Clear cache or rerun to re-upload.")

        question = st.text_input("Ask about your chat history")
        model_name = st.selectbox(
            "Model",
            ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=1,
        )
        if st.button("Ask", type="primary") and question.strip():
            if not api_key:
                st.error("Set GEMINI_API_KEY first.")
            else:
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=question,
                        config=genai_types.GenerateContentConfig(
                            tools=[
                                genai_types.Tool(
                                    file_search=genai_types.FileSearch(
                                        file_search_store_names=[store_name]
                                    )
                                )
                            ]
                        ),
                    )
                    st.markdown("**Answer**")
                    st.write(getattr(resp, "text", "") or "(No text returned)")
                except Exception as e:
                    st.error(f"Query failed: {e}")



if __name__ == "__main__":
    main()
