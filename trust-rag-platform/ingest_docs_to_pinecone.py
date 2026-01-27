import os
import glob
import hashlib
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trust-rag-platform")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "internal-docs-v1")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Your folder name (with space)
DOCS_DIR = "data"

# Simple deterministic chunking (good for demos)
CHUNK_SIZE = 2000   # chars
OVERLAP = 200       # chars


def chunk_text(text: str):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - OVERLAP)
    return chunks


def stable_id(source: str, chunk_index: int, chunk: str) -> str:
    # stable id so reruns don't create duplicates
    h = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:10]
    return f"{source}#c{chunk_index:03d}#{h}"


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    # Assumes index dimension matches the chosen embedding model
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def main():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found. Ensure it's set in your .env file.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found. Required to embed text before upserting.")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    oa_client = OpenAI(api_key=openai_api_key)

    pattern = os.path.join(DOCS_DIR, "*.md")
    md_files = sorted(glob.glob(pattern))

    if not md_files:
        raise FileNotFoundError(
            f"No .md files found under ./{DOCS_DIR}\n"
            f"Expected pattern: {pattern}\n"
            f"Tip: run `ls \"{DOCS_DIR}\"` to confirm file extensions."
        )

    payloads = []
    for fp in md_files:
        source = os.path.basename(fp)

        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        if not chunks:
            print(f"⚠️ Skipping empty file: {source}")
            continue

        for i, ch in enumerate(chunks):
            rec_id = stable_id(source, i, ch)

            payloads.append({
                "id": rec_id,
                "text": ch,
                "metadata": {
                    "source": source,
                    "chunk_id": f"c{i:03d}",
                    "char_len": len(ch),
                },
            })

    print(f"📦 Found {len(md_files)} files. Uploading {len(payloads)} chunks to Pinecone...")

    embeddings = embed_texts(oa_client, [p["text"] for p in payloads])

    vectors = []
    for payload, emb in zip(payloads, embeddings):
        meta = dict(payload["metadata"])
        meta["text"] = payload["text"]
        vectors.append({
            "id": payload["id"],
            "values": emb,
            "metadata": meta,
        })

    index.upsert(namespace=NAMESPACE, vectors=vectors)

    print("✅ Ingestion complete.")
    print("📊 Index stats:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
