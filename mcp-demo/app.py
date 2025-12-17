import json
import requests
import streamlit as st

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def build_prompt(system_prompt: str, history: list[dict], user_input: str) -> str:
    """
    Convert chat messages into a single prompt for /api/generate.
    """
    parts = []
    if system_prompt.strip():
        parts.append(f"System:\n{system_prompt.strip()}\n")

    for m in history:
        role = m["role"].capitalize()
        parts.append(f"{role}:\n{m['content']}\n")

    parts.append(f"User:\n{user_input}\n")
    parts.append("Assistant:\n")
    return "\n".join(parts)


def ollama_generate_stream(model: str, prompt: str):
    """
    Streams text from Ollama /api/generate and yields chunks.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("response", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


st.set_page_config(page_title="Local LLM Agent (Ollama)", layout="wide")
st.title("🧠 Local LLM Agent (Ollama + Streamlit)")
st.caption("Chat locally with your Ollama model via /api/generate.")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    system_prompt = st.text_area(
        "System instructions (optional)",
        value="You are a helpful assistant. Keep answers clear and practical.",
        height=120,
    )
    st.markdown("---")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask anything…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = build_prompt(system_prompt, st.session_state.messages[:-1], user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        collected = ""

        try:
            for chunk in ollama_generate_stream(model=model, prompt=prompt):
                collected += chunk
                placeholder.markdown(collected)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach Ollama. Make sure it's running on 127.0.0.1:11434.")
            st.stop()
        except requests.HTTPError as e:
            st.error(f"Ollama HTTP error: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": collected})