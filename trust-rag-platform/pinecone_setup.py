import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trust-rag-platform")
# OpenAI text-embedding-3-small produces 1536-dim vectors; override via env if needed.
INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION", "1536"))
INDEX_METRIC = os.getenv("PINECONE_INDEX_METRIC", "cosine")

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is required.")

    pc = Pinecone(api_key=api_key)

    existing = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Created index: {INDEX_NAME}")
    else:
        print(f"ℹ️ Index already exists: {INDEX_NAME}")

if __name__ == "__main__":
    main()
