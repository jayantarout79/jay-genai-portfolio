from __future__ import annotations

import datetime as dt
import pathlib
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st

# Allow running `streamlit run app.py` from inside the package directory.
if __package__ in (None, ""):
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from prompt_caching_lab.config import OPENAI_MODEL, get_price_sheet
from prompt_caching_lab.metrics_store import add_run, get_runs_df, init_store
from prompt_caching_lab.openai_client import call_model
from prompt_caching_lab.prompt_logic import (
    build_long_prompt,
    estimate_token_length,
    explain_prompt_structure,
)

st.set_page_config(
    page_title="OpenAI Prompt Caching Lab",
    page_icon="🧠",
    layout="wide",
)

# Keep run history alive across Streamlit reruns.
init_store()


def compute_cost(usage: Dict[str, float], model: str) -> float:
    """Approximate cost for a single run using configurable price sheet."""
    prices = get_price_sheet(model)
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    billable_prompt = max(prompt_tokens - cached_tokens, 0)
    cost = (
        (billable_prompt / 1000) * prices.prompt_per_1k
        + (cached_tokens / 1000) * prices.cached_prompt_per_1k
        + (completion_tokens / 1000) * prices.completion_per_1k
    )
    return round(cost, 6)


def run_experiment(user_question: str, repetitions: int, model: str) -> str:
    """Run repeated calls with the same messages to surface caching effects."""
    messages = build_long_prompt(user_question)
    last_answer = ""
    for _ in range(repetitions):
        try:
            answer, usage = call_model(messages, model=model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"OpenAI call failed: {exc}")
            break

        last_answer = answer
        # Persist per-run metrics for display.
        add_run(
            {
                "timestamp": dt.datetime.now(dt.UTC),
                "user_question": user_question,
                "latency_seconds": usage.get("latency_seconds", 0.0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "cached_tokens": usage.get("cached_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "model": model,
                "cache_hit": usage.get("cache_hit", False),
            }
        )
    return last_answer


def render_metrics(model: str) -> None:
    """Render tables, charts, and totals."""
    df = get_runs_df()
    if df.empty:
        st.info("No runs yet. Configure a prompt in the sidebar and click 'Run experiment'.")
        return

    df = df.copy()
    # Add helpful display columns.
    df["run"] = range(1, len(df) + 1)
    df["timestamp_short"] = df["timestamp"].dt.strftime("%H:%M:%S")
    df["cache_status"] = df["cache_hit"].apply(lambda hit: "Hit" if hit else "Miss")

    # Compute approximate costs per run
    df["estimated_cost"] = df.apply(lambda row: compute_cost(row.to_dict(), model), axis=1)

    cols = [
        "run",
        "timestamp_short",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "estimated_cost",
        "cache_status",
    ]
    st.subheader("Run metrics")
    st.dataframe(df[cols], hide_index=True, use_container_width=True)

    total_cached = int(df["cached_tokens"].sum())
    total_prompt = int(df["prompt_tokens"].sum())
    total_hits = int(df["cache_hit"].sum())
    total_cost = df["estimated_cost"].sum()

    st.metric("Total cached tokens", f"{total_cached:,}")
    st.metric("Total prompt tokens", f"{total_prompt:,}")
    st.metric("Approximate total cost (USD)", f"${total_cost:.6f}")
    st.metric("Cache hits", f"{total_hits} / {len(df)} runs")
    st.caption("Costs are approximate; update pricing in config.py as needed.")

    if total_cached == 0:
        st.warning(
            "Cached tokens are zero so far. Ensure the system prompt stays the same, "
            "has a long prefix (>1024 tokens), and the selected model supports caching."
        )

    chart_df = df[["run", "latency_seconds", "cached_tokens"]].set_index("run")

    st.subheader("Latency by run")
    st.line_chart(chart_df["latency_seconds"], height=240)

    st.subheader("Cached tokens by run")
    st.bar_chart(chart_df["cached_tokens"], height=240)


def main() -> None:
    st.title("OpenAI Prompt Caching Lab")
    st.write(
        "Run the same long prompt multiple times to see how OpenAI's prompt caching "
        "reduces prompt token usage and latency on subsequent calls."
    )

    # Sidebar: basic inputs for the experiment.
    with st.sidebar:
        st.header("Configure experiment")
        user_question = st.text_area(
            "User question",
            value="Summarize the three biggest challenges and how AI can help.",
            height=120,
        )
        model_options = ["gpt-5.1", "gpt-5", "gpt-4.1"]
        model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(OPENAI_MODEL) if OPENAI_MODEL in model_options else 0,
        )
        repetitions = st.slider("Number of repeated runs", min_value=1, max_value=10, value=3)
        run_button = st.button("Run experiment", type="primary")

    # Preview the prompt that will be sent.
    messages_preview = build_long_prompt(user_question)
    system_preview = messages_preview[0]["content"]
    user_preview = messages_preview[1]["content"]
    est_tokens = estimate_token_length(system_preview) + estimate_token_length(user_preview)

    st.subheader("Prompt preview")
    st.write(explain_prompt_structure())
    st.caption(f"Estimated tokens (rough): {est_tokens}")
    st.write("System prompt (first 400 chars):")
    st.code(system_preview[:400] + "...", language="text")
    with st.expander("Full system prompt"):
        st.write(system_preview)

    if run_button:
        last_answer = run_experiment(user_question, repetitions, model)
        if last_answer:
            st.success(f"Completed {repetitions} run(s).")
            with st.expander("Sample model response"):
                st.write(last_answer)

    # Metrics and charts for recent runs.
    render_metrics(model)


if __name__ == "__main__":
    main()
