import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

from rag_guardrails import RetrievalMatch

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trust-rag-platform")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "internal-docs-v1")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

pc_api_key = os.getenv("PINECONE_API_KEY")
oa_api_key = os.getenv("OPENAI_API_KEY")

if not pc_api_key:
    raise RuntimeError("PINECONE_API_KEY is required.")
if not oa_api_key:
    raise RuntimeError("OPENAI_API_KEY is required.")

pc = Pinecone(api_key=pc_api_key)
index = pc.Index(INDEX_NAME)
oa_client = OpenAI(api_key=oa_api_key)


def retrieve(query_text: str, top_k: int = 5) -> List[RetrievalMatch]:
    # Embed the query locally; the index is configured for vector search (no integrated inference).
    embedding = oa_client.embeddings.create(
        model=EMBED_MODEL,
        input=query_text
    ).data[0].embedding

    res = index.query(
        namespace=NAMESPACE,
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )

    matches: List[RetrievalMatch] = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) or {}
        matches.append(
            RetrievalMatch(
                score=float(m["score"]),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", "unknown"),
            )
        )
    return matches
