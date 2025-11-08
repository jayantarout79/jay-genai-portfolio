# app.py — Gemini File Search UI (simple)
import os
import time
import pandas as pd
import streamlit as st
import tempfile
from pathlib import Path

# pip install -U google-genai python-dotenv
from google import genai
from google.genai import types
from dotenv import load_dotenv

st.set_page_config(page_title="Gemini File Search — Simple UI", layout="wide")
st.title("🔎 Gemini File Search — Managed RAG (Simple UI)")

# -----------------------------
# Env / API Key
# -----------------------------
load_dotenv()  # load GEMINI_API_KEY if present

def set_api_key(k: str):
    if k:
        os.environ["GEMINI_API_KEY"] = k

with st.sidebar:
    st.subheader("API Key")
    default_key = os.getenv("GEMINI_API_KEY", "")
    key = st.text_input("GEMINI_API_KEY", type="password", value=default_key, help="Stored only for this session.")
    if st.button("Save key", type="primary", use_container_width=True):
        set_api_key(key)
        st.success("API key set for this session.")

# Guard: client
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    st.info("Add your GEMINI_API_KEY in the sidebar to start.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_client(k: str):
    return genai.Client(api_key=k)

client = get_client(api_key)

# -----------------------------
# Create File Search Store
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_store(_client: genai.Client):
    # Keep it minimal to match the working sample
    store = _client.file_search_stores.create()
    return store.name  # e.g. fileSearchStores/abc123

try:
    store_name = create_store(client)
    st.caption(f"Using File Search Store: `{store_name}`")
except Exception as e:
    st.error(f"File Search Store could not be created. {e}")
    st.stop()

st.divider()

# -----------------------------
# Upload & Index
# -----------------------------
st.subheader("1) Upload files to index")

uploaded_files = st.file_uploader(
    "Drop PDF, DOCX, TXT, MD, CSV (<=10MB each).",
    type=["pdf", "docx", "txt", "md", "csv"],
    accept_multiple_files=True,
)

if st.button("Upload & Index", type="primary") and uploaded_files:
    for f in uploaded_files:
        try:
            # Write the uploaded file to a temporary path so the SDK can infer mime/type from the filename
            tmp_dir = tempfile.mkdtemp(prefix="gfs_")
            tmp_path = Path(tmp_dir) / f.name
            with open(tmp_path, "wb") as out:
                out.write(f.getbuffer())

            # Upload by path (string) – avoids mime_type issues with file-like objects
            upload_name = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store_name,
                file=str(tmp_path),
            )

            # Poll until done
            with st.spinner(f"Indexing {f.name} …"):
                start = time.time()
                while True:
                    op = client.operations.get(upload_name)
                    if getattr(op, "done", False):
                        break
                    if time.time() - start > 180:
                        raise TimeoutError("Indexing took too long.")
                    time.sleep(1.5)
            st.success(f"Indexed ✅ {f.name}")
        except Exception as e:
            st.error(f"Upload/index failed for {f.name}: {e}")

# -----------------------------
# List documents
# -----------------------------
st.subheader("Indexed documents")
docs_df = None
try:
    docs_iter = client.file_search_stores.documents.list(
        file_search_store_name=store_name, page_size=200
    )
    docs = list(docs_iter)
    if docs:
        docs_df = pd.DataFrame([
            {
                "id": d.name,
                "name": getattr(d, "display_name", getattr(d, "filename", "")) or "",
                "size_bytes": getattr(d, "size_bytes", 0) or 0,
                "state": getattr(d, "state", "ACTIVE") or "ACTIVE",
            }
            for d in docs
        ])
except Exception as e:
    st.warning(f"Could not list documents: {e}")

if docs_df is not None:
    st.dataframe(docs_df, hide_index=True, use_container_width=True)
else:
    st.caption("No files indexed yet.")

st.divider()

# -----------------------------
# Ask Question (grounded to store)
# -----------------------------
st.subheader("2) Ask your knowledge base")
question = st.text_input("Your question", placeholder="e.g., What are the total sales by region?")
model_name = st.selectbox(
    "Model",
    ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    index=1,
)

if st.button("Ask", type="primary", use_container_width=False) and question.strip():
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ]
            ),
        )
        st.markdown("**Answer**")
        st.write(getattr(resp, "text", "") or "(No text returned)")

        # Citations (grounding)
        try:
            grounding = resp.candidates[0].grounding_metadata
            titles = {c.retrieved_context.title for c in grounding.grounding_chunks}
            if titles:
                st.markdown("**Citations**")
                for t in titles:
                    st.markdown(f"- {t}")
            else:
                st.caption("No explicit citations returned.")
        except Exception:
            st.caption("No explicit citations returned.")
    except Exception as e:
        st.error(f"Query failed: {e}")
