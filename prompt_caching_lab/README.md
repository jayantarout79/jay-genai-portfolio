# Prompt Caching Lab

A minimal Streamlit app that shows how OpenAI prompt caching reduces prompt token billing and latency when you reuse the same long system prompt. The app uses an Astra system prompt (fixed), lets you choose a model, and repeats the same request to surface cache hits.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your key and preferred model
streamlit run prompt_caching_lab/app.py
```

## Configuration

Set these environment variables:

- `OPENAI_API_KEY` – required.
- `OPENAI_MODEL` – optional; defaults to `gpt-5.1`. You can also pick `gpt-5`, or `gpt-4.1` in the UI.

Pricing numbers in `config.py` are scaled from OpenAI’s published per-million token rates and used only for rough estimates.

## How it works

- The Astra system prompt stays constant to maximize cache reuse; only your user question varies.
- Each run calls `chat.completions` and records:
  - `prompt_tokens`, `completion_tokens`, `total_tokens`
  - `prompt_tokens_details.cached_tokens` to detect cache hits
  - request latency (seconds)
- Costs are estimated per run using the price sheet in `config.py`.

## Using the app

1. Enter a short **user question**.
2. Pick a **model** (`gpt-5.1`, `gpt-5`, or `gpt-4.1`).
3. Choose **repetitions** (e.g., 3–5) and click **Run experiment**.
4. Watch the metrics table and charts: cache hits should appear on later runs, and latency often drops.

## Notes and limitations

- Cache behavior is opaque and may vary by request or model.
- Numbers are approximate; update pricing as OpenAI changes rates.
- Demo only; not intended for billing-grade accounting or SLA commitments.
