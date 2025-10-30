from google.adk.agents.llm_agent import Agent
from pinecone import Pinecone
from openai import OpenAI
import os

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.getenv("PINECONE_INDEX", "rag-demo-openai-embed"))

def _embed(text: str) -> list[float]:
    # OpenAI embeddings (can swap to Gemini if you prefer)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def pinecone_retrieval(query: str, top_k: int = 5):
    vec = _embed(query)

    # New-style query (returns matches with metadata)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        namespace="__default__"
    )

    # Normalize a simple return payload for the agent
    hits = []
    for m in res.get("matches", []):
        hits.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "metadata": m.get("metadata", {})
        })

    return {"results": hits}

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction=(
        "You are a retrieval-only AI assistant. "
        "Answer strictly and only using information returned in tool results "
        "from the Pinecone vector database. Do not invent or guess. "
        "If the answer is not found in the retrieved results, reply with: "
        "'I’m sorry, I don’t have information about that in my current knowledge base.' "
        "When you answer, include which source or category you used from metadata."
    ),
    description="Retrieval-Augmented Generation agent that queries Pinecone for context.",
    tools=[pinecone_retrieval]
)