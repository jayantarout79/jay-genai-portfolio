# src/chunker.py
"""
Chunk raw documents into overlapping text segments for RAG indexing.

Usage:
  python src/chunker.py data --out index/chunks.jsonl --size 1000 --overlap 200

Notes:
- Relies on loaders.load_documents() to return a list of LangChain Documents.
- Uses RecursiveCharacterTextSplitter for robust, structure-aware chunking.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional

# --- LangChain splitters (v0.2+ first, fallback to older import) ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:  # pragma: no cover
    raise ImportError(
        "Could not import RecursiveCharacterTextSplitter from langchain. "
        "Please ensure that 'langchain' is installed and up to date."
    )

# allow running this file directly: python src/chunker.py ...
import sys
sys.path.append(os.path.dirname(__file__))

from langchain_core.documents import Document  # LC document type

# local
from loaders import load_documents


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Create a robust character splitter with sensible separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", "\n",  # paragraphs / lines
            "### ", "## ", "# ",  # markdown headers
            ". ", "? ", "! ",     # sentence-ish
            ", ", "; ", " ",      # words / spaces
            ""                    # fallback
        ],
        add_start_index=True,
    )


def _chunk_one_doc(doc: Document, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Split a single Document and add chunk-level metadata."""
    chunks = splitter.split_documents([doc])
    total = len(chunks)
    # enrich metadata
    out: List[Document] = []
    for i, ch in enumerate(chunks):
        md = dict(doc.metadata) if doc.metadata else {}
        md.update({
            "chunk_index": i,
            "chunk_total": total,
        })
        out.append(Document(page_content=ch.page_content, metadata=md))
    return out


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Chunk a list of Documents and return the expanded list."""
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    all_chunks: List[Document] = []
    for d in docs:
        all_chunks.extend(_chunk_one_doc(d, splitter))
    return all_chunks


def save_jsonl(chunks: List[Document], out_path: str) -> None:
    """Write chunks to JSONL for easy inspection / later indexing."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in chunks:
            rec: Dict[str, Any] = {
                "text": d.page_content,
                "metadata": d.metadata or {},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def preview_stats(chunks: List[Document]) -> None:
    """Print quick stats to sanity-check chunking."""
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        print("[WARN] No chunks produced.")
        return
    print(f"[INFO] Chunks: {len(chunks)}  |  min/mean/max chars: "
          f"{min(lengths)}/{int(sum(lengths)/len(lengths))}/{max(lengths)}")
    # show a couple of samples
    for idx in (0, len(chunks)//2, len(chunks)-1):
        if 0 <= idx < len(chunks):
            md = chunks[idx].metadata
            src = md.get("source", "unknown")
            ci = md.get("chunk_index", "?")
            ct = md.get("chunk_total", "?")
            print(f"\n--- sample #{idx}  ({src}  chunk {ci+1}/{ct}) ---")
            print(chunks[idx].page_content[:300].strip(), "...")


def main():
    parser = argparse.ArgumentParser(description="Chunk documents for RAG indexing")
    parser.add_argument("data_dir", help="Path to folder with source files (md/pdf/…)")
    parser.add_argument("--out", default="index/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--size", type=int, default=int(os.getenv("CHUNK_SIZE", 1000)), help="Chunk size")
    parser.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", 200)), help="Chunk overlap")
    args = parser.parse_args()

    # 1) load raw docs
    docs = load_documents(args.data_dir)
    print(f"[INFO] Loaded {len(docs)} documents from {args.data_dir}")

    # 2) chunk
    chunks = chunk_documents(docs, chunk_size=args.size, chunk_overlap=args.overlap)

    # 3) preview + save
    preview_stats(chunks)
    save_jsonl(chunks, args.out)
    print(f"\n[OK] Saved chunks → {args.out}")


if __name__ == "__main__":
    main()