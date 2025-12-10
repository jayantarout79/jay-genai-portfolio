from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from batch_utils import (
    build_batch_jsonl,
    create_batch_job,
    download_batch_output_file,
    ensure_data_dirs,
    get_batch_status,
    get_client,
    load_feedback_csv,
    merge_results_with_input,
    parse_batch_output,
    upload_batch_file,
)


st.set_page_config(page_title="OpenAI Batch Job Studio", layout="wide")

# Ensure required folders exist on startup.
DATA_DIR, SAMPLE_DIR = ensure_data_dirs()


def init_state() -> None:
    """Initialize commonly used session state keys."""
    defaults = {
        "df": None,
        "data_path": None,
        "batch_id": None,
        "input_file_id": None,
        "output_file_id": None,
        "error_file_id": None,
        "model": "gpt-4.1-mini",
        "last_submitted_at": None,
        "results_df": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

st.title("OpenAI Batch Job Studio")
st.caption(
    "Upload a CSV, build a JSONL, send a Batch job to the Responses API, then come back to fetch results asynchronously."
)

# Sidebar configuration
st.sidebar.header("Batch configuration")
selected_model = st.sidebar.selectbox(
    "Model", options=["gpt-4.1-mini", "gpt-4.1"], index=0
)
max_rows = st.sidebar.number_input("Max rows to send", min_value=1, max_value=200, value=50)


def load_sample_data() -> None:
    """Load bundled sample CSV into session state."""
    try:
        df = load_feedback_csv(SAMPLE_DIR / "feedback.csv")
        st.session_state.df = df
        st.session_state.data_path = SAMPLE_DIR / "feedback.csv"
        st.sidebar.success(f"Loaded {len(df)} sample rows.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to load sample data: {exc}")


if st.sidebar.button("Load sample data"):
    load_sample_data()

if st.session_state.df is not None:
    st.sidebar.write(f"Rows loaded: **{len(st.session_state.df)}**")
    st.sidebar.write(f"Source: `{st.session_state.data_path}`")


def handle_file_upload(upload: Optional[object]) -> None:
    """
    Handle user CSV upload, validating columns and storing in session_state.
    """
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
        required_cols = {"id", "customer_text"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            return
        st.session_state.df = df
        st.session_state.data_path = Path(upload.name)
        st.success(f"Uploaded {len(df)} rows from {upload.name}.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read uploaded CSV: {exc}")


# Step 1 – Submit Batch Job
st.markdown("---")
st.subheader("Step 1 – 🚀 Submit Batch Job")
st.write("Upload a CSV or use the sample data, then submit a Batch job to classify all feedback rows.")

uploaded = st.file_uploader("Upload your CSV (columns: id, customer_text)", type=["csv"])
handle_file_upload(uploaded)

if st.session_state.df is not None:
    st.dataframe(st.session_state.df.head(5), use_container_width=True)
else:
    st.info("Load sample data from the sidebar or upload your own CSV to begin.")


def submit_batch_job() -> None:
    """Build JSONL, upload, and submit the batch job."""
    if st.session_state.df is None:
        st.error("No data loaded. Please upload a CSV or load the sample data.")
        return
    df = st.session_state.df.head(int(max_rows)).copy()
    model = selected_model
    jsonl_path = DATA_DIR / "batch_input.jsonl"
    try:
        build_batch_jsonl(df, model=model, output_path=jsonl_path)
        client = get_client()
        input_file_id = upload_batch_file(client, jsonl_path)
        batch_id = create_batch_job(client, file_id=input_file_id)
        st.session_state.batch_id = batch_id
        st.session_state.input_file_id = input_file_id
        st.session_state.model = model
        st.session_state.last_submitted_at = time.time()
        st.success(
            f"Batch submitted! Batch ID: `{batch_id}`. Keep this ID to check status in Step 2."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to submit batch: {exc}")


if st.button("Build JSONL and submit batch", type="primary"):
    with st.spinner("Submitting batch job..."):
        submit_batch_job()


# Step 2 – Check Status
st.markdown("---")
st.subheader("Step 2 – 📊 Check Status")
st.write("Refresh the batch status. Completed batches will expose an output file for download.")

if not st.session_state.batch_id:
    st.info("Submit a batch first to get a Batch ID.")
else:
    st.code(st.session_state.batch_id, language="text")

    def refresh_status() -> None:
        try:
            client = get_client()
            status = get_batch_status(client, st.session_state.batch_id)
            st.session_state.output_file_id = status.get("output_file_id")
            st.session_state.error_file_id = status.get("error_file_id")
            st.session_state.request_counts = status.get("request_counts", {})
            st.session_state.status = status.get("status")
            badge_color = "green" if status["status"] == "completed" else "orange"
            if status["status"] == "failed":
                badge_color = "red"
            st.markdown(f"**Status:** <span style='color:{badge_color}'>{status['status']}</span>", unsafe_allow_html=True)
            counts = status.get("request_counts", {})
            st.write(
                pd.DataFrame(
                    {
                        "total": [counts.get("total", 0)],
                        "completed": [counts.get("completed", 0)],
                        "failed": [counts.get("failed", 0)],
                    }
                )
            )
            st.write(f"Input file ID: `{status.get('input_file_id')}`")
            if status.get("output_file_id"):
                st.success("Output file is ready. Proceed to Step 3.")
                st.write(f"Output file ID: `{status.get('output_file_id')}`")
            if status.get("error_file_id"):
                st.warning(f"Error file ID: `{status.get('error_file_id')}`")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not refresh status: {exc}")

    if st.button("Refresh status"):
        with st.spinner("Retrieving batch status..."):
            refresh_status()


# Step 3 – View Results
st.markdown("---")
st.subheader("Step 3 – ✅ View Results")
st.write("Download the completed batch output, parse it, and explore the merged results.")

if not st.session_state.get("output_file_id"):
    st.info("Batch is not completed yet or output file not available.")
else:

    def download_and_parse() -> None:
        output_path = DATA_DIR / "batch_output.jsonl"
        try:
            client = get_client()
            download_batch_output_file(client, st.session_state.output_file_id, output_path)
            parsed = parse_batch_output(output_path)
            merged = merge_results_with_input(st.session_state.df, parsed)
            st.session_state.results_df = merged
            st.success("Results downloaded and parsed.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to download or parse results: {exc}")

    if st.button("Download and parse results"):
        with st.spinner("Downloading and parsing output..."):
            download_and_parse()

    results_df = st.session_state.get("results_df")
    if results_df is None or results_df.empty or "sentiment" not in results_df.columns:
        st.info("No parsed results yet. Ensure the batch has completed and returned output.")

    if results_df is not None and not results_df.empty:
        total_rows = len(results_df)
        if "sentiment" in results_df.columns:
            sentiment_counts = results_df["sentiment"].value_counts(dropna=False)
        else:
            sentiment_counts = pd.Series(dtype=int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total rows", total_rows)
        c2.metric("Positive", int(sentiment_counts.get("positive", 0)))
        c3.metric("Neutral", int(sentiment_counts.get("neutral", 0)))
        c4.metric("Negative", int(sentiment_counts.get("negative", 0)))

        st.dataframe(results_df, use_container_width=True)

        if not sentiment_counts.empty:
            st.bar_chart(sentiment_counts)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download merged CSV",
            data=csv_bytes,
            file_name="batch_results.csv",
            mime="text/csv",
        )
    else:
        st.info("No results parsed yet. Download the output file first.")
