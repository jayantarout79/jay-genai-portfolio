import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

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

query_text = "What happens when a pipeline fails?"

# Embed query locally; index is configured for vector search (no integrated inference).
embedding = oa_client.embeddings.create(
    model=EMBED_MODEL,
    input=query_text
).data[0].embedding

res = index.query(
    namespace=NAMESPACE,
    vector=embedding,
    top_k=5,
    include_metadata=True,
    include_values=False,
)

print("Query:", query_text)
for i, m in enumerate(res["matches"], 1):
    meta = m.get("metadata", {})
    print(f"{i}. score={m['score']:.4f} source={meta.get('source')} chunk={meta.get('chunk_id')}")
