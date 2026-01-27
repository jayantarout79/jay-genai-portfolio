# Trust RAG Platform

## Overview
- Trust-first RAG prototype that only answers when retrieval evidence is sufficient.
- Stack: Pinecone (vector DB), OpenAI (embeddings + chat), Streamlit UI, Python utilities for ingestion and smoke tests.

## Setup
1) Python 3.10+ recommended.  
2) Install deps: `pip install -r requirement.txt`.  
3) Create `.env` with keys:  
   - `PINECONE_API_KEY=`  
   - `OPENAI_API_KEY=`  
   - Optional: `PINECONE_INDEX_NAME` (default `trust-rag-platform`), `PINECONE_NAMESPACE` (default `internal-docs-v1`), `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`), `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`).

## Workflow
- Create/verify index: `python3 pinecone_setup.py` (sets dimension/metric for the embedding model).  
- Ingest docs: `python3 ingest_docs_to_pinecone.py` (reads `data/*.md`, chunks, embeds via OpenAI, upserts vectors+metadata).  
- Smoke test query: `python3 query_smoketest.py`.  
- CLI demo: `python3 rag_demo_cli.py`.  
- Streamlit app: `streamlit run app.py`.

## Files of interest
- `pinecone_setup.py` — creates Pinecone serverless index with env-driven name/dimension/metric.  
- `ingest_docs_to_pinecone.py` — chunks `data/*.md`, embeds, and upserts to Pinecone.  
- `rag_retrieve.py` — embeds query and queries Pinecone (vector search).  
- `rag_guardrails.py` — decision gates (answer/abstain), confidence signals, citations.  
- `rag_demo_cli.py` — terminal demo using guardrails.  
- `app.py` — Streamlit UI with retrieval, guardrails, grounded answer via OpenAI chat.  
- `data/` — markdown source documents.

## Notes / Troubleshooting
- Ensure Pinecone index dimension matches the embedding model (e.g., text-embedding-3-small → 1536). If you change models, recreate the index with the matching dimension.  
- Errors about “Integrated inference” mean you’re using `search`; stick to `index.query` with explicit embeddings as in current code.  
- If retrieval returns low scores, guardrails will abstain; add more relevant docs or adjust chunking.  
- Logs from the Streamlit app are written to `logs/` when enabled.
