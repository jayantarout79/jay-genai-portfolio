# streamlit_app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import re
import difflib

# --- Path bootstrap so absolute imports work when run via `streamlit run` ---
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]  # points to jay-genai-portfolio
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faiss_query_engine import FaissQueryEngine
from faiss_rag_demo import answer_with_csv

# ---------- Setup ----------
load_dotenv()  # allows OPENAI_API_KEY from .env for local runs

st.set_page_config(page_title="Sales Q&A (FAISS + RAG)", layout="wide")
st.title("🧠 Sales Q&A — FAISS + RAG (Practice)")

# ---------- Sidebar Controls ----------
st.sidebar.header("Data & Index Settings")

# Default to your Superstore CSV; let user change it
default_csv = "/Users/jkr/Desktop/FutureGoal/pythonprojects/jay-genai-portfolio/Simple-Python/SampleSuperstore.csv"
csv_path = st.sidebar.text_input("CSV path", value=default_csv)

# Choose columns for embeddings (start with sensible defaults)
default_cols = ["City", "State", "Category", "Sub-Category", "ProductName", "Sales", "Quantity"]
cols_text = st.sidebar.text_input("Embedded columns (comma-separated)", value=", ".join(default_cols))
embedded_columns = [c.strip() for c in cols_text.split(",") if c.strip()]

k = st.sidebar.slider("Top-K rows to retrieve", min_value=5, max_value=200, value=50, step=5)
rebuild = st.sidebar.checkbox("Force rebuild index (once)", value=False)

# (Optional) let user override models via env/secrets if they want
embed_model = st.sidebar.text_input("Embeddings model", value="text-embedding-3-small")
llm_model = st.sidebar.text_input("LLM model (used in your RAG demo script)", value="gpt-4o-mini")

st.sidebar.divider()
st.sidebar.caption("Make sure OPENAI_API_KEY is set via .env or Streamlit Secrets.")

# ---------- Caching: engine + dataframe ----------
@st.cache_resource(show_spinner=True)
def get_engine(_csv_path: str, _cols: tuple, _embed_model: str, _rebuild: bool):
    # We instantiate once per unique parameter combo
    eng = FaissQueryEngine(
        csv_path=_csv_path,
        embedded_columns=list(_cols),
        embedding_model=_embed_model,
        rebuild=_rebuild
    )
    return eng

@st.cache_data(show_spinner=False)
def get_dataframe(_engine: FaissQueryEngine) -> pd.DataFrame:
    return _engine.df.copy()

# ---------- Analytics helper (simple, optional) ----------
AGG_KEYWORDS = ("most", "highest", "top", "sum", "total", "count", "max")

def _extract_after_in(q: str) -> str:
    """Extract the phrase after the word 'in' for location matching; fallback to full query."""
    m = re.search(r"\bin\s+([a-zA-Z\s\-]+)", q)
    return m.group(1).strip() if m else q

def _find_place_and_column(query: str, df_full: pd.DataFrame):
    """
    Fuzzy-match a place name from the query against City/State values in the dataframe.
    Returns (matched_value, column_name) or (None, None).
    """
    if df_full.empty:
        return None, None
    q = query.lower()
    segment = _extract_after_in(q).lower()

    city_vals = []
    state_vals = []
    if "City" in df_full.columns:
        city_vals = list(df_full["City"].dropna().astype(str).str.lower().unique())
    if "State" in df_full.columns:
        state_vals = list(df_full["State"].dropna().astype(str).str.lower().unique())
    candidates = city_vals + state_vals
    if not candidates:
        return None, None

    # Use close matches against the extracted segment
    matches = difflib.get_close_matches(segment, candidates, n=1, cutoff=0.6)
    if not matches:
        # fall back to scanning tokens within the whole query
        tokens = re.findall(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)?", q)
        tokens = [" ".join(t.split()).strip() for t in tokens]
        matches = difflib.get_close_matches(" ".join(tokens), candidates, n=1, cutoff=0.6)
        if not matches:
            return None, None

    best = matches[0]
    if best in city_vals:
        return best, "City"
    if best in state_vals:
        return best, "State"
    return None, None

def try_analytics_answer(query: str, df_full: pd.DataFrame):
    """
    Handles deterministic analytics-style questions without LLM hallucination,
    e.g., 'total sales in florida', 'sum of sales in newywork (sic)', 'top 5 by quantity in texas'.
    - Fuzzy-matches city/state from the query
    - Supports SUM/COUNT/MAX intents and 'most/top/highest' phrasing
    """
    q = query.lower()
    place_val, place_col = _find_place_and_column(q, df_full)

    # --- Simple "total/sum of sales ..." path (no ranking words required) ---
    sum_like = ("total" in q) or ("sum" in q) or ("sales" in q and not any(w in q for w in AGG_KEYWORDS))
    if sum_like and "Sales" in df_full.columns:
        df = df_full.copy()
        if place_val and place_col in df.columns:
            df = df[df[place_col].astype(str).str.lower() == place_val]
        if df.empty:
            return None
        total_sales = float(df["Sales"].sum())
        pretty_place = f" in {place_val.title()}" if place_val else ""
        return f"Total Sales{pretty_place}: {total_sales:.2f}"

    # --- "most/top/highest" style ranking using Quantity or Sales ---
    if not any(w in q for w in AGG_KEYWORDS):
        return None

    df = df_full.copy()
    if df.empty:
        return None

    if place_val and place_col in df.columns:
        df = df[df[place_col].astype(str).str.lower() == place_val]
    if df.empty:
        return None

    metric = "Quantity" if "Quantity" in df.columns else ("Sales" if "Sales" in df.columns else None)
    by = "ProductName" if "ProductName" in df.columns else None
    if metric is None or by is None:
        return None

    ans = (
        df.groupby(by, dropna=False)[metric]
          .sum()
          .sort_values(ascending=False)
          .head(5)
    )
    header = f"Top 5 by {metric}"
    if place_val:
        header += f" in {place_val.title()}"
    lines = "\n".join(f"- {k}: {v}" for k, v in ans.items())
    return header + ":\n" + lines

# ---------- Main UI ----------
st.subheader("Ask a question about the sales data")
query = st.text_input("Your question", value="Most furniture sold in New York")

col_run, col_about = st.columns([1, 3])
with col_run:
    run = st.button("Run")

# Show preview
if csv_path and Path(csv_path).exists():
    st.caption("Dataset preview")
    try:
        eng = get_engine(csv_path, tuple(embedded_columns), embed_model, rebuild)
        df_full = get_dataframe(eng)
        st.dataframe(df_full.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load engine or data: {e}")
        st.stop()
else:
    st.error("CSV path not found. Please correct the path in the sidebar.")
    st.stop()

# ---------- Execute query ----------
if run:
    with st.spinner("Retrieving relevant rows..."):
        try:
            # Analytics fallback for “most/top” questions
            analytic = try_analytics_answer(query, df_full)
            df_matches = eng.semantic_search(query, k=k)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    # Show matches table
    if df_matches.empty:
        st.warning("No relevant rows found.")
    else:
        keep_cols = [c for c in ["City","State","Category","Sub-Category","ProductName","Sales","Quantity","score"] if c in df_matches.columns]
        st.subheader("Top matches")
        st.dataframe(df_matches[keep_cols].head(k), use_container_width=True)

    st.subheader("Answer")
    if analytic:
        # Deterministic, correct for “most/top” style questions
        st.write(analytic)
    else:
        # If you want to add LLM reasoning over the rows (RAG-style),
        # call your faiss_rag_demo.answer_with_csv here.
        # For now, give a grounded short summary directly from matches.
        if df_matches.empty:
            st.info("I don't know.")
        else:
            # Simple grounded response without LLM:
            top_names = df_matches["ProductName"].head(5).tolist() if "ProductName" in df_matches.columns else []
            st.write("I found these most relevant items:")
            st.write(", ".join(top_names) if top_names else "Matches shown above.")

with col_about:
    st.caption("""
**How it works**  
- Uses your existing `FaissQueryEngine` to persist/load a FAISS index over selected columns.  
- Retrieves top‑K rows for the query and shows them as citations.  
- For “most/top/sum” style questions, a tiny pandas groupby gives a deterministic answer.  
- (Optional) Add your LLM step to summarize/compose an answer using retrieved rows.
""")