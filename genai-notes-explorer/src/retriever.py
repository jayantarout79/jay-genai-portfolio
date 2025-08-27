# src/retriever.py
"""
Query the FAISS index and return top-k chunks with scores.

Usage:
  python3 src/retriever.py --index index/ --query "What is PageValues and why does it matter?" --k 5
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_index(outdir: str) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    index_path = os.path.join(outdir, "faiss.index")
    meta_path = os.path.join(outdir, "meta.jsonl")
    model_path = os.path.join(outdir, "model.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index at {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata at {meta_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model manifest at {model_path}")

    index = faiss.read_index(index_path)
    meta = list(read_jsonl(meta_path))
    with open(model_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, meta, manifest

def get_client_and_model(model_arg: str | None = None) -> Tuple[OpenAI, str]:
    load_dotenv()
    client = OpenAI()
    model = model_arg or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return client, model

def embed_query(client: OpenAI, model: str, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    # normalize for cosine via inner product
    faiss.normalize_L2(vec)
    return vec

def search(index: faiss.Index, meta: List[Dict[str, Any]], qvec: np.ndarray, k: int = 5):
    k = min(k, index.ntotal)
    scores, ids = index.search(qvec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        rec = meta[idx]
        results.append({
            "score": float(score),
            "text": rec["text"],
            "metadata": rec.get("metadata", {})
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="FAISS retriever")
    parser.add_argument("--index", default="index", help="Directory containing faiss.index/meta.jsonl/model.json")
    parser.add_argument("--query", required=True, help="User question / query")
    parser.add_argument("--k", type=int, default=int(os.getenv("RETRIEVAL_K", 5)))
    parser.add_argument("--model", default=None, help="Embedding model (overrides env)")
    args = parser.parse_args()

    index, meta, manifest = load_index(args.index)
    client, embed_model = get_client_and_model(args.model)
    print(f"[INFO] Loaded index with {index.ntotal} vectors (dim={index.d})")
    print(f"[INFO] Using embedding model: {embed_model}")

    qvec = embed_query(client, embed_model, args.query)
    results = search(index, meta, qvec, k=args.k)

    print("\n[TOP RESULTS]")
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "unknown")
        ci = r["metadata"].get("chunk_index", "?")
        print(f"\n{i}) score={r['score']:.4f} | {src}#chunk{ci}")
        # trim preview
        snippet = r["text"].strip().replace("\n", " ")
        print(snippet[:300] + ("..." if len(snippet) > 300 else ""))

if __name__ == "__main__":
    main()