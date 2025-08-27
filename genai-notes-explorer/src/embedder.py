# src/embedder.py
"""
Build and persist a FAISS index from chunked documents (chunks.jsonl).

Usage:
  python3 src/embedder.py build --chunks index/chunks.jsonl --out index/ \
      --model text-embedding-3-small --batch 128
  python3 src/embedder.py stats --out index/

Outputs (in --out):
  - faiss.index        : FAISS vector index
  - meta.jsonl         : one JSON per vector ({"text": ..., "metadata": ...})
  - model.json         : {"embedding_model": "...", "dim": 1536, "count": N}
"""

import os
import json
import time
import math
import argparse
from typing import List, Dict, Any, Iterator, Tuple

import numpy as np

# FAISS
import faiss  # pip install faiss-cpu

# OpenAI v1 SDK
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------
# Helpers
# ---------------------------

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def batched(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


# ---------------------------
# OpenAI Embeddings
# ---------------------------

def get_client_and_model(model_arg: str | None = None) -> Tuple[OpenAI, str]:
    load_dotenv()
    client = OpenAI()  # reads OPENAI_API_KEY from env
    model = model_arg or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return client, model


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """
    Returns a (N, D) float32 NumPy array of embeddings for the given texts.
    Uses simple batching with basic retry/backoff.
    """
    all_vecs: List[np.ndarray] = []
    for batch in batched(texts, batch_size):
        # Retry loop (basic)
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
                all_vecs.append(np.vstack(vecs))
                break
            except Exception as e:
                wait = 1.5 * (attempt + 1)
                print(f"[WARN] Embedding batch failed ({e}), retrying in {wait:.1f}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Failed to embed after retries.")
    return np.vstack(all_vecs)


# ---------------------------
# FAISS Index build/save/load
# ---------------------------

def build_index_from_embeddings(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds an Inner Product (cosine-ready) index.
    Note: Normalize vectors for cosine similarity.
    """
    # Normalize to unit length for cosine similarity via dot product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(outdir: str, index: faiss.IndexFlatIP, meta_records: List[Dict[str, Any]], model: str) -> None:
    ensure_outdir(outdir)
    # Save FAISS index
    faiss.write_index(index, os.path.join(outdir, "faiss.index"))
    # Save meta aligned to index vectors
    write_jsonl(os.path.join(outdir, "meta.jsonl"), meta_records)
    # Save simple manifest
    dim = index.d
    write_json(os.path.join(outdir, "model.json"), {
        "embedding_model": model,
        "dim": dim,
        "count": index.ntotal
    })


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
    manifest = json.load(open(model_path, "r", encoding="utf-8"))
    return index, meta, manifest


def build_faiss_index(chunks_path: str, out_dir: str, model: str | None = None, batch_size: int = 128) -> None:
    """
    Compatibility wrapper used by src/cli.py. Reads a chunks.jsonl file,
    embeds texts, builds a FAISS IP (cosine-normalized) index and persists it
    alongside aligned metadata.
    """
    # 1) Read chunks
    chunks = list(read_jsonl(chunks_path))
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    texts = [c["text"] for c in chunks]
    metas = [c.get("metadata", {}) for c in chunks]

    # 2) Client + model
    client, model_name = get_client_and_model(model)

    # 3) Embed
    vecs = embed_texts(client, model_name, texts, batch_size=batch_size)

    # 4) Build index (cosine-ready)
    index = build_index_from_embeddings(vecs)

    # 5) Persist
    meta_records = [
        {"text": t, "metadata": m}
        for t, m in zip(texts, metas)
    ]
    save_index(out_dir, index, meta_records, model_name)


def build_index_from_chunks(chunks_path: str, out_dir: str, model: str | None = None, batch_size: int = 128) -> None:
    """Alias for backward compatibility."""
    return build_faiss_index(chunks_path, out_dir, model, batch_size)


# ---------------------------
# Build command
# ---------------------------

def cmd_build(chunks_path: str, outdir: str, model: str | None, batch: int) -> None:
    # 1) Read chunks
    chunks = list(read_jsonl(chunks_path))
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    texts = [c["text"] for c in chunks]
    metas = [c.get("metadata", {}) for c in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks from {chunks_path}")

    # 2) Client + model
    client, model_name = get_client_and_model(model)
    print(f"[INFO] Using embedding model: {model_name}")

    # 3) Embed
    vecs = embed_texts(client, model_name, texts, batch_size=batch)
    print(f"[INFO] Embedded shape: {vecs.shape} (N x D)")

    # 4) Build index
    index = build_index_from_embeddings(vecs)
    print(f"[INFO] FAISS index built with {index.ntotal} vectors (dim={index.d})")

    # 5) Persist
    # Keep text + metadata together so retriever can cite easily
    meta_records = [
        {"text": t, "metadata": m}
        for t, m in zip(texts, metas)
    ]
    save_index(outdir, index, meta_records, model_name)
    print(f"[OK] Saved FAISS index + metadata to: {outdir}")


# ---------------------------
# Stats command
# ---------------------------

def cmd_stats(outdir: str) -> None:
    index, meta, manifest = load_index(outdir)
    print("[INFO] Index stats")
    print(f"  - vectors : {index.ntotal}")
    print(f"  - dim     : {index.d}")
    print(f"  - model   : {manifest.get('embedding_model')}")
    print(f"  - metas   : {len(meta)} records")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Build / inspect FAISS index from chunked docs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build FAISS index from chunks.jsonl")
    b.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    b.add_argument("--out", default="index", help="Output directory for index files")
    b.add_argument("--model", default=None, help="Embedding model name (overrides env)")
    b.add_argument("--batch", type=int, default=128, help="Embedding batch size")

    s = sub.add_parser("stats", help="Show index stats")
    s.add_argument("--out", default="index", help="Directory with faiss.index")

    args = parser.parse_args()

    if args.cmd == "build":
        cmd_build(args.chunks, args.out, args.model, args.batch)
    elif args.cmd == "stats":
        cmd_stats(args.out)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()