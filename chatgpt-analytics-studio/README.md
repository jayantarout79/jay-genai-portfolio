# ChatGPT Analytics Studio

Interactive Streamlit app that processes a ChatGPT export, surfaces early usage metrics, runs OpenAI/Gemini analysis on user-only prompts, and renders a neon-style dashboard (plus persona image) ready for sharing.

## Features
- One-click pipeline: ingest export, filter to user prompts, chunk conversations, and scan audio/image artifacts.
- Early Insights: KPIs (messages, chars, distinct words, audio minutes, images), messages-per-month trend, top words, and longest words (with labels).
- AI Insights: per-chunk topics, prompt quality scoring, sentiment, and prompt-quality breakdown.
- AI Final Insights: overall summary plus highlights, risks, recommendations, and improvement tips (clean bullet formatting).
- Sentiment Curve: month-wise sentiment with highest/lowest sample prompts.
- Nano Bana Profile: generates a high-res persona/dashboard image via Gemini/OpenAI images.
- Chat tab: Gemini File Search over indexed chunks so you can ask questions against your own history (uploads cached by fingerprint per session).

## Tech Stack
- UI: Streamlit, Altair (charts with inline labels), custom CSS for neon styling.
- Data: pandas, wordfreq, mutagen/wave for audio durations.
- AI: OpenAI chat/image, Google Gemini (genai) for file search + image gen.
- State/Caching: local JSON state, chunked JSON files, minimal .env loader, Streamlit cache.

## Flow (high level)
1) Ingest/export check → fingerprint raw folder.
2) Early insights → user-only messages, counts, words, monthly series, audio/image scans.
3) Preprocess → chunk user prompts, create placeholder audio transcripts.
4) AI analysis → per-chunk topics/sentiment/prompt quality (OpenAI) and aggregates.
5) Final insights → summarize highlights/risks/recs/tips; persona/dashboard image generation.
6) Sentiment curve → month-wise sentiment; most positive/negative samples.
7) Chat → upload chunk JSONs to Gemini File Search store once per session and query.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set secrets in `.env` (not committed):
```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OPENAI_IMAGE_MODEL=gpt-image-1   # optional override
```

## Run
```bash
streamlit run app.py
```

Use “Run from start” for the full pipeline. In the Chat tab, click “Index chunk files to store” once per dataset (cached by fingerprint) before asking questions. Images require `GEMINI_API_KEY` or OpenAI image support.

## Notes for Git
- `.gitignore` excludes raw exports, generated chunks/transcripts, aggregates, and secrets.
- All charts render value labels; final insight bullets are cleaned of brackets/colons.
- If you change the export, rerun preprocessing/analysis and reindex chunks for chat.
