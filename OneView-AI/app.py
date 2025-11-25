"""Streamlit entry point for the Multimodal Video → Analytics Engine."""

from __future__ import annotations

from pathlib import Path
import logging
import os
import time
import smtplib
from email.message import EmailMessage
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv
import streamlit as st

from ai import insights
from ai.llm_client import LLMClient
from components.navigation import vertical_nav
from core import documents, transcript, video_io
from core import did_client
from theme.style import inject_global_styles
from ui import layout

MAX_FILE_SIZE_MB = 300
SUPPORTED_TYPES = ["mp4", "mov", "m4v", "csv", "pdf", "txt", "log", "jpg", "jpeg", "png"]
NAV_ITEMS = [
    "Overview",
    "Summary & Actions",
]

logging.basicConfig(level=logging.INFO)
load_dotenv()


def init_state() -> None:
    st.session_state.setdefault("video_state", {})
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    st.session_state.setdefault("did_api_key", os.getenv("D_ID_API_KEY", ""))
    st.session_state.setdefault("transcript_df", pd.DataFrame())
    st.session_state.setdefault("ai_last_called", None)
    st.session_state.setdefault("active_nav", NAV_ITEMS[0])
    st.session_state.setdefault("uploaded_assets", [])
    st.session_state.setdefault("processing_state", "")
    st.session_state.setdefault("multimodal_bundle", {})
    st.session_state.setdefault("did_talk", {})
    st.session_state.setdefault("flow_status", {})
    st.session_state.setdefault("email_recipient", os.getenv("SMTP_TO", ""))


def handle_file_upload(uploaded_files) -> None:
    """Persist uploaded files and classify them for processing."""
    if not uploaded_files:
        return
    assets = st.session_state.get("uploaded_assets", [])
    for uploaded_file in uploaded_files:
        if uploaded_file.size and uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB} MB limit.")
            continue
        temp_path = video_io.persist_uploaded_file(uploaded_file)
        kind = documents.classify_kind(Path(uploaded_file.name))
        assets.append({"path": str(temp_path), "name": uploaded_file.name, "kind": kind})
        if kind == "video":
            st.session_state["video_state"] = {
                "video_path": str(temp_path),
                "video_name": uploaded_file.name,
            }
            st.session_state["chat_history"] = []
    st.session_state["uploaded_assets"] = assets


def process_video(llm_client: LLMClient) -> None:
    if not llm_client.available:
        st.error("OpenAI API key required before processing.")
        return
    video_state = st.session_state.get("video_state", {})
    video_path = video_state.get("video_path")
    if not video_path:
        st.warning("Please upload a video before processing.")
        return

    try:
        update_processing_state("Extracting audio track...")
        audio_path, duration = video_io.extract_audio(Path(video_path))
        video_state["audio_path"] = str(audio_path)
        video_state["duration"] = duration

        update_processing_state("Transcribing audio via Whisper...")
        segments = transcript.transcribe_audio(Path(video_state["audio_path"]), llm_client)
        transcript_df = transcript.segments_to_dataframe(segments)
        video_state["transcript_df"] = transcript_df
        st.session_state["transcript_df"] = transcript_df

        st.session_state["video_state"] = video_state
        st.session_state["chat_history"] = []
    except Exception as exc:  # pragma: no cover - streamlit runtime feedback
        logging.exception("Processing failed: ")
        st.error(f"Processing failed: {exc}")


def update_processing_state(state: str) -> None:
    st.session_state["processing_state"] = state


def update_flow(step: str, status: str) -> None:
    flow = st.session_state.get("flow_status", {})
    flow[step] = status
    st.session_state["flow_status"] = flow


def process_multimodal(llm_client: LLMClient, did_api_key: str, source_image_url: str) -> None:
    """Run the full pipeline across video + auxiliary files."""
    uploads = st.session_state.get("uploaded_assets", [])
    if not uploads:
        st.warning("Upload at least one file to begin processing.")
        return
    st.session_state["did_talk"] = {}
    update_processing_state("Starting ingestion and checks...")
    update_flow("upload", "in_progress")
    update_flow("data", "pending")
    update_flow("video", "pending")
    update_flow("insights", "pending")
    update_flow("avatar", "pending")
    update_flow("delivery", "pending")
    try:
        descriptors, text_blobs = documents.ingest_files(uploads)
        st.session_state["uploaded_assets"] = descriptors
        update_flow("upload", "complete")
        st.toast("Files ingested", icon="✅")
        if any(item["kind"] == "video" for item in descriptors):
            update_processing_state("Transcribing video and running analytics...")
            update_flow("data", "in_progress")
            process_video(llm_client)
            update_flow("data", "complete")
            update_flow("video", "complete")
            st.toast("Video transcription done", icon="✅")

        transcript_df = st.session_state.get("transcript_df", pd.DataFrame())
        update_processing_state("Generating insights and action items with AI...")
        update_flow("insights", "in_progress")
        bundle = insights.generate_multimodal_bundle(text_blobs, transcript_df, llm_client)
        st.session_state["multimodal_bundle"] = bundle
        update_flow("insights", "complete")
        st.toast("Insights generated", icon="✅")

        script = bundle.get("narration_script") or "Here is your meeting recap and next steps."
        if did_api_key and source_image_url:
            update_processing_state("Sending narration to D-ID...")
            update_flow("avatar", "in_progress")
            try:
                talk_id = did_client.create_talk(did_api_key, script, source_image_url)
                st.session_state["did_talk"] = {"id": talk_id, "status": "created"}
                # Poll until ready
                for _ in range(6):
                    time.sleep(5)
                    status = did_client.fetch_talk(did_api_key, talk_id)
                    talk_status = status.get("status")
                    st.session_state["did_talk"] = {
                        "id": talk_id,
                        "status": talk_status,
                        "result_url": status.get("result_url"),
                    }
                    if talk_status == "done" and status.get("result_url"):
                        update_flow("avatar", "complete")
                        st.toast("Avatar video ready", icon="✅")
                        break
                if st.session_state["did_talk"].get("status") != "done":
                    update_flow("avatar", "pending")
            except Exception as exc:  # pragma: no cover - external service guard
                logging.exception("D-ID generation failed")
                st.error(f"D-ID generation failed: {exc}")
        update_processing_state("Processing completed")
        update_flow("delivery", "ready")
        st.toast("Delivery ready", icon="✅")
    except Exception as exc:  # pragma: no cover - runtime guard
        logging.exception("Multimodal processing failed")
        st.error(f"Processing failed: {exc}")
        update_processing_state("Failed. See logs.")


def refresh_did_status() -> None:
    talk_info = st.session_state.get("did_talk") or {}
    api_key = st.session_state.get("did_api_key") or ""
    talk_id = talk_info.get("id")
    if not api_key or not talk_id:
        st.info("No D-ID request to refresh.")
        return
    try:
        status = did_client.fetch_talk(api_key, talk_id)
        st.session_state["did_talk"] = {
            "id": talk_id,
            "status": status.get("status"),
            "result_url": status.get("result_url"),
        }
    except Exception as exc:  # pragma: no cover - external service guard
        logging.exception("Failed to refresh D-ID status")
        st.error(f"Could not refresh D-ID status: {exc}")


def send_email_stub() -> None:
    bundle = st.session_state.get("multimodal_bundle") or {}
    summary = bundle.get("email_summary") or "AI summary not generated yet."
    insights = bundle.get("insights") or []
    actions = bundle.get("actions") or []
    recipient = st.session_state.get("email_recipient") or os.getenv("SMTP_TO") or os.getenv("SMTP_USER")
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM") or user
    video_url = (st.session_state.get("did_talk") or {}).get("result_url")
    if not all([host, user, password, sender, recipient]):
        st.error("Email not sent. Configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO in .env.")
        return
    insight_list = "\n".join(f"- {i}" for i in insights) if insights else "No insights generated."
    action_list = "\n".join(
        f"- {a.get('item', 'Action')} (Owner: {a.get('owner', 'Unknown')}, Due: {a.get('due', 'TBD')})"
        for a in actions
    ) if actions else "No actions generated."
    body = (
        "Executive Summary\n\n"
        f"{summary}\n\n"
        "Key Insights:\n"
        f"{insight_list}\n\n"
        "Recommended Actions:\n"
        f"{action_list}\n\n"
        "Video:\n"
        f"{video_url or 'Pending (rendering)'}\n"
    )
    msg = EmailMessage()
    msg["Subject"] = "OneView AI Insights & Actions"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(body)
    if video_url:
        msg.add_alternative(
            f"<p><strong>Executive Summary</strong></p><p>{summary}</p>"
            f"<p><strong>Key Insights</strong><br>{'<br>'.join(insights)}</p>"
            f"<p><strong>Recommended Actions</strong><br>{'<br>'.join(action_list.splitlines())}</p>"
            f"<p><a href='{video_url}'>Download Avatar Video</a></p>",
            subtype="html",
        )
    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        st.success(f"Email sent to {recipient}.")
    except Exception as exc:  # pragma: no cover - runtime guard
        logging.exception("Email send failed")
        st.error(f"Email send failed: {exc}")


def _status_pill(label: str, sub: str, tone: str = "neutral") -> str:
    tone_map = {
        "good": "#0f9d58",
        "warn": "#f97316",
        "neutral": "#9aa5b1",
    }
    dot_color = tone_map.get(tone, "#9aa5b1")
    return (
        f"<div class='status-pill'><div class='dot' style='background:{dot_color}'></div>"
        f"<div><div class='label'>{label}</div><div class='sub'>{sub}</div></div></div>"
    )


def main() -> None:
    st.set_page_config(page_title="Multimodal Video → Analytics Engine", page_icon="🎬", layout="wide")
    inject_global_styles(dark=False)

    init_state()

    st.markdown(
        "<div style='text-align:center; padding: 0.4rem 0 0.8rem 0;'>"
        "<h1 style='margin-bottom:4px;'>OneView AI</h1>"
        "<p style='margin-top:0;color:var(--muted-color);font-size:16px;'>All your data. One clear story.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    nav_col, content_col = st.columns([0.7, 3.3], gap="large")
    with nav_col:
        st.markdown("#### Pipeline & Delivery")
        stored_key = os.getenv("OPENAI_API_KEY", "")
        st.session_state["openai_api_key"] = stored_key
        llm_client = LLMClient(api_key=stored_key or None)

        st.markdown("#### Delivery")
        stored_did_key = os.getenv("D_ID_API_KEY", "")
        st.session_state["did_api_key"] = stored_did_key
        source_image_url = os.getenv("D_ID_SOURCE_URL", "")

        status_html = [
            _status_pill(
                "AI Status",
                "Ready" if llm_client.available else "Set OPENAI_API_KEY in .env",
                "good" if llm_client.available else "warn",
            ),
            _status_pill(
                "D-ID",
                "Configured" if stored_did_key else "Add D_ID_API_KEY",
                "good" if stored_did_key else "warn",
            ),
            _status_pill(
                "Uploads",
                f"{len(st.session_state.get('uploaded_assets', []))} files",
                "neutral",
            ),
        ]
        st.markdown("<div class='status-row'>" + "".join(status_html) + "</div>", unsafe_allow_html=True)

        st.markdown("#### Upload")
        uploads = st.file_uploader(
            "Browse files",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        handle_file_upload(uploads)

        uploaded_assets = st.session_state.get("uploaded_assets", [])
        if uploaded_assets:
            chips = "".join(f"<span class='chip'>{item['name']} · {item['kind']}</span>" for item in uploaded_assets[-8:])
            st.markdown(f"<div class='upload-chips'>{chips}</div>", unsafe_allow_html=True)

        disabled = not llm_client.available
        st.button(
            "Run Multimodal Pipeline",
            type="primary",
            disabled=disabled or not uploaded_assets,
            on_click=process_multimodal,
            args=(llm_client, stored_did_key, source_image_url),
            key="process_multimodal_btn",
            use_container_width=True,
        )

        # suppress sidebar AI status callout

    video_state = st.session_state.get("video_state", {})
    transcript_df: pd.DataFrame = st.session_state.get("transcript_df", pd.DataFrame())

    with content_col:
        active_section = st.radio(
            "Navigate",
            NAV_ITEMS,
            index=NAV_ITEMS.index(st.session_state.get("active_nav", NAV_ITEMS[0])),
            horizontal=True,
        )
        st.session_state["active_nav"] = active_section
        layout.render_processing_banner(st.session_state.get("processing_state", ""))
        layout.render_upload_preview(st.session_state.get("uploaded_assets", []))
        layout.render_flow_progress(st.session_state.get("flow_status", {}), st.session_state.get("did_talk", {}))
        layout.render_delivery_panel(
            st.session_state.get("multimodal_bundle", {}),
            st.session_state.get("did_talk", {}),
            on_email=send_email_stub,
        )
        if active_section == "Overview":
            layout.render_overview_tab(video_state, transcript_df)
        elif active_section == "Summary & Actions":
            layout.render_summary_tab(st.session_state.get("multimodal_bundle", {}))


if __name__ == "__main__":
    main()
