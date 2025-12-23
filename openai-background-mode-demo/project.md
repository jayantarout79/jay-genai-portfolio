# LLM Background Mode Lab

Single-page Streamlit dashboard demonstrating the OpenAI Responses API in three modes: synchronous, background (polling), and background + streaming.

## Features
- Prompt, model, and mode selection (default model `gpt-5.2`).
- Synchronous calls for quick replies.
- Background jobs with polling, auto-refresh, and cancel support.
- Background streaming with live token updates, cursor tracking, and reconnect placeholder.
- Session-state persistence for IDs, statuses, timestamps, buffers, and events.
- Raw response JSON and stream event log expanders.
- Notes on response expiration and ZDR caveat.

## Project Layout
- `app.py` — Streamlit app with all logic and UI.
- `.env` — Local env vars (ignored by git); set `OPENAI_API_KEY`.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Git ignore rules for env, caches, and Streamlit artifacts.

## Prerequisites
- Python 3.9+
- OpenAI API key with access to chosen models.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your key (preferred in `.env`):
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

## Run
```bash
streamlit run app.py
```

## Usage Notes
- Submit triggers the selected mode; submit is disabled on empty prompt.
- Auto-refresh runs until a terminal status is reached.
- Cancel is idempotent for queued/in-progress jobs.
- Streaming loop only runs during submission; subsequent updates rely on stored state and refresh controls.
- If streaming reconnect is attempted, a placeholder message is shown (SDK resume unsupported here). Polling remains available.
