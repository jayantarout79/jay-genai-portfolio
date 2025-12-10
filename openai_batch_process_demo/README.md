# OpenAI Batch Job Studio

A small, hands-on Streamlit app that demonstrates the OpenAI Batch API by classifying customer feedback in bulk. It walks through the full async flow: upload CSV → build JSONL → submit batch → wait → download results.

## What this shows
- OpenAI Batch API with `/v1/responses`
- Asynchronous job submission and retrieval
- JSONL input files for batch processing
- Bulk sentiment + category classification and summarization

## Setup
1. Clone the repo.
2. Create a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy `.env.example` to `.env` and set your real `OPENAI_API_KEY`.
5. Run the app: `streamlit run app.py`.

## How to use
1. Step 1: Load the bundled sample data from the sidebar or upload your CSV, then submit a batch job.
2. Step 2: Wait a bit, then refresh the batch status until it shows `completed`.
3. Step 3: Download the batch output, parse it, and explore the merged results in the dashboard or export as CSV.

## Cost notes
- Batch API requests are typically cheaper than synchronous calls (about 50% in many cases).
- Use the "Max rows" control to keep demo runs small and affordable.

## Ideas for extensions
- Swap the Responses call for embeddings or other endpoints.
- Schedule nightly batch runs for new feedback.
- Add email or Slack notifications when batches finish.
