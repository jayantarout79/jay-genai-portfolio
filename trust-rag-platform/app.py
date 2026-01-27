import os
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

from rag_guardrails import (
    RetrievalMatch,
    decide_answer_or_abstain,
    build_citations,
    suggest_clarifying_question,
)

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trust-rag-platform")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "internal-docs-v1")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(page_title="Trust RAG Platform", layout="wide")

st.title("Trust RAG Platform")
st.caption("A trust-first RAG prototype that answers only when evidence is sufficient.")

# ---------- Pinecone client ----------
p_api_key = os.getenv("PINECONE_API_KEY")
if not p_api_key:
    st.error("PINECONE_API_KEY not found in .env")
    st.stop()

o_api_key = os.getenv("OPENAI_API_KEY")
if not o_api_key:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

pc = Pinecone(api_key=p_api_key)
index = pc.Index(INDEX_NAME)
oa_client = OpenAI(api_key=o_api_key)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K", min_value=3, max_value=10, value=5, step=1)
show_signals = st.sidebar.checkbox("Show confidence signals", value=True)
show_raw_matches = st.sidebar.checkbox("Show raw matches", value=False)
enable_logging = st.sidebar.checkbox("Log queries (local)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Index: **{INDEX_NAME}**")
st.sidebar.caption(f"Namespace: **{NAMESPACE}**")

# ---------- Main input ----------
q = st.text_input("Ask a question", placeholder="e.g., What happens when a pipeline fails?")

colA, colB = st.columns([1, 1])
ask = colA.button("Ask", use_container_width=True)
clear = colB.button("Clear", use_container_width=True)

if clear:
    st.session_state.clear()
    st.rerun()

def pinecone_retrieve(query_text: str, k: int):
    # Embed locally; the index stores vectors (no integrated inference).
    embedding = oa_client.embeddings.create(
        model=EMBED_MODEL,
        input=query_text
    ).data[0].embedding

    res = index.query(
        namespace=NAMESPACE,
        vector=embedding,
        top_k=k,
        include_metadata=True,
        include_values=False,
    )

    matches = []
    for m in res["matches"]:
        meta = m.get("metadata", {}) or {}
        matches.append(
            RetrievalMatch(
                score=float(m["score"]),
                text=meta.get("text", ""),   # requires ingestion to store metadata["text"]
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", "unknown"),
            )
        )
    return matches

def log_event(payload: dict):
    if not enable_logging:
        return
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(LOG_DIR, f"rag_log_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# ---------- Helper: LLM answer grounded in citations ----------
def generate_grounded_answer_openai(question: str, citations: list) -> str:
    """
    Craft a clean answer grounded ONLY in the retrieved citations.
    If evidence is insufficient, the model must say it doesn't know.
    """
    evidence_blocks = []
    for i, c in enumerate(citations, 1):
        evidence_blocks.append(
            f"[{i}] source={c['source']}:{c['chunk_id']} score={c['score']}\n{c.get('evidence_text', c['snippet'])}\n"
        )
    evidence = "\n".join(evidence_blocks).strip()

    system = (
        "You are a careful enterprise data platform assistant. "
        "Answer the question using ONLY the EVIDENCE provided. "
        "If the evidence does not contain the answer, reply exactly: "
        "\"I don't know based on the provided documents.\" "
        "Do not guess or add external knowledge. "
        "Be concise and structured."
    )

    user = f"""QUESTION:
{question}

EVIDENCE:
{evidence}

RESPONSE FORMAT:
- Answer (2-5 sentences)
- Citations: include bracket numbers like [1], [2]
"""

    resp = oa_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

if ask and q.strip():
    query = q.strip()

    with st.spinner("Retrieving evidence from Pinecone..."):
        matches = pinecone_retrieve(query, top_k)

    decision = decide_answer_or_abstain(matches, user_query=query)
    citations = build_citations(matches, max_citations=min(3, len(matches)))

    # ---------- Decision banner ----------
    if decision.action == "ANSWER":
        st.success("✅ ANSWER (Evidence sufficient)")
    else:
        st.warning("⚠️ ABSTAIN (Evidence insufficient)")

    st.write(f"**Reason:** {decision.reason}")

    if show_signals:
        st.subheader("Confidence Signals")
        st.json(decision.signals)

    # ---------- Evidence / citations ----------
    st.subheader("Retrieved Evidence (Top Matches)")
    for i, c in enumerate(citations, 1):
        with st.expander(f"{i}) {c['source']}:{c['chunk_id']}  |  score={c['score']}"):
            st.write(c["snippet"])

    # ---------- Output ----------
    st.subheader("Output")
    if decision.action == "ABSTAIN":
        st.info("I don’t have enough evidence in the provided documents to answer this reliably.")
        st.write("**Suggested clarifying question:**")
        st.write(suggest_clarifying_question(query))
    else:
        if not citations:
            st.info("No citations were available to ground an answer.")
        else:
            with st.spinner("Crafting a grounded answer (OpenAI)..."):
                final_answer = generate_grounded_answer_openai(query, citations)
            st.write(final_answer)

    if show_raw_matches:
        st.subheader("Raw Matches")
        st.write([{"score": m.score, "source": m.source, "chunk_id": m.chunk_id} for m in matches])

    # ---------- Logging ----------
    log_event({
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "decision": decision.action,
        "reason": decision.reason,
        "signals": decision.signals,
        "top_matches": [
            {"score": m.score, "source": m.source, "chunk_id": m.chunk_id}
            for m in matches[:top_k]
        ],
    })
else:
    st.markdown("### Demo tips")
    st.markdown(
        "- Try an answerable query: **What happens when a pipeline fails?**\n"
        "- Try an abstain query: **What is the SLA for all pipelines?**\n"
        "- The goal is to show **trust-first behavior**, not just fluent answers."
    )
