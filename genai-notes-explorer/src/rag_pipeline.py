# src/rag_pipeline.py
"""
Usage:
  python3 src/rag_pipeline.py --index index/ --query "What is PageValues and why does it matter?" --k 5
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

from prompts import SYSTEM_PROMPT, USER_PROMPT
from llm_client import chat_complete

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_index(outdir: str):
    index = faiss.read_index(os.path.join(outdir, "faiss.index"))
    meta = list(read_jsonl(os.path.join(outdir, "meta.jsonl")))
    with open(os.path.join(outdir, "model.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, meta, manifest

def embed_query(text: str, model: str) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def retrieve(index, meta, qvec, k: int = 5):
    k = min(k, index.ntotal)
    scores, ids = index.search(qvec, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        rec = meta[idx]
        out.append({
            "score": float(score),
            "text": rec["text"],
            "metadata": rec.get("metadata", {})
        })
    return out

def format_context(results: List[Dict[str, Any]]) -> str:
    """Join top chunks and append a citation tag to each block."""
    blocks = []
    for r in results:
        md = r["metadata"]
        src = md.get("source", "unknown")
        chunk_id = md.get("chunk_index", "?")
        tag = f"[source:{src}#chunk{chunk_id}]"
        blocks.append(f"{r['text'].strip()}\n{tag}")
    return "\n\n---\n\n".join(blocks)

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG pipeline: retrieve → prompt → LLM")
    parser.add_argument("--index", default="index/", help="FAISS index directory")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=int(os.getenv("RETRIEVAL_K", "5")))
    parser.add_argument("--embed_model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--chat_model", default=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    index, meta, _ = load_index(args.index)

    qvec = embed_query(args.query, args.embed_model)
    results = retrieve(index, meta, qvec, k=args.k)
    context = format_context(results)

    user = USER_PROMPT.format(question=args.query, context=context)
    answer = chat_complete(SYSTEM_PROMPT, user, model=args.chat_model)

    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES ===")
    for i, r in enumerate(results, 1):
        md = r["metadata"]
        print(f"{i}. {md.get('source','unknown')}#chunk{md.get('chunk_index','?')}  (score={r['score']:.4f})")

if __name__ == "__main__":
    main()