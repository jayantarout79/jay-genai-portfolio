import streamlit as st

from agent import run_research_agent

st.set_page_config(page_title="Local Research Agent", layout="wide")

st.title("Local Research Agent (LangChain + Web + OpenAI)")
st.caption("Type a topic → fetch web sources → generate a ~500 word brief with citations.")

with st.sidebar:
    st.header("Settings")
    max_results = st.slider("Web results", min_value=3, max_value=10, value=5)
    use_cache = st.checkbox("Use local cache", value=True)

topic = st.text_input("Topic", placeholder="e.g., Retrieval-Augmented Generation for enterprise search")
go = st.button("Generate brief", type="primary")

if go:
    try:
        with st.spinner("Researching and writing..."):
            result = run_research_agent(topic, use_cache=use_cache, max_results=max_results)

        st.subheader("Research Brief")
        st.write(result.brief)

        st.subheader("Sources")
        for i, s in enumerate(result.sources, start=1):
            title = s.get("title") or f"Source {i}"
            url = s.get("url") or ""
            st.markdown(f"**[{i}] {title}**")
            if url:
                st.write(url)

    except Exception as e:
        st.error(str(e))