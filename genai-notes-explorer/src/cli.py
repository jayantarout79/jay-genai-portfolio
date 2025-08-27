# src/cli.py
"""
One-file CLI for your Notes RAG:
- rebuild: load -> chunk -> embed -> save FAISS index
- ask    : retrieve -> prompt -> LLM answer (with citations)

Examples:
  python3 src/cli.py rebuild \
    --data data --chunks index/chunks.jsonl --out index \
    --size 1000 --overlap 200 --embed_model text-embedding-3-small --batch 128

  python3 src/cli.py ask \
    --index index --query "What metrics did we optimize and why?" \
    --k 5 --embed_model text-embedding-3-small --chat_model gpt-4o-mini
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

# --- import helpers so we can run from repo root or from src/ ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# --- dotenv (optional) ---
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()
except Exception:
    pass

# --- local modules ---
from loaders import load_documents                  # src/loaders.py
from chunker import chunk_documents, save_jsonl     # src/chunker.py

# strictly import the positional API from embedder
try:
    from embedder import build_faiss_index, cmd_stats as print_index_stats  # src/embedder.py
except Exception as e:
    raise RuntimeError(f"[cli] Could not import from embedder: {e}")

from rag_pipeline import (                          # src/rag_pipeline.py
    load_index,
    embed_query,
    retrieve,
    format_context,
    chat_complete,
)

# Optional: prompt templates
try:
    from rag_pipeline import SYSTEM_PROMPT, USER_PROMPT
except Exception:
    SYSTEM_PROMPT = (
        "You are a precise assistant. Answer with bullet-first, grounded in the provided context. "
        "If evidence is weak, say so."
    )
    USER_PROMPT = (
        "Question:\n{question}\n\n"
        "Context (use for citations):\n{context}\n\n"
        "Answer in concise bullets, and include inline [source:file#chunk] markers."
    )


def env_or(key: str, default: Any) -> Any:
    v = os.getenv(key)
    return type(default)(v) if v is not None else default


# tolerant wrapper: embedder.build_faiss_index may accept (chunks_path, out_dir, model, batch_size)
# or a single config dict {chunks_path, out_dir, model, batch_size} depending on your local embedder.py
def _call_build_index(chunks_path: str, out_dir: str, model: str, batch_size: int) -> None:
    try:
        # try the 4-positional-args signature first
        return build_faiss_index(chunks_path, out_dir, model, batch_size)  # type: ignore[misc]
    except TypeError:
        # fall back to single-config signature
        cfg = {
            "chunks_path": chunks_path,
            "out_dir": out_dir,
            "model": model,
            "batch_size": batch_size,
        }
        return build_faiss_index(cfg)  # type: ignore[misc]


# ----------------- subcommand: rebuild -----------------
def cmd_rebuild(args: argparse.Namespace) -> None:
    data_dir: Path = Path(args.data)
    out_chunks: Path = Path(args.chunks)
    out_dir: Path = Path(args.out)

    if not data_dir.exists():
        raise FileNotFoundError(f"[rebuild] data folder not found: {data_dir}")

    print(f"[rebuild] Loading documents from: {data_dir}")
    docs = load_documents(str(data_dir))
    print(f"[rebuild] Loaded {len(docs)} documents")

    print(f"[rebuild] Chunking (size={args.size}, overlap={args.overlap}) …")
    chunks = chunk_documents(docs, chunk_size=args.size, chunk_overlap=args.overlap)

    out_chunks.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(chunks, str(out_chunks))
    print(f"[rebuild] Saved chunks → {out_chunks}")

    print(f"[rebuild] Embedding with model={args.embed_model}, batch={args.batch} …")
    # IMPORTANT: call embedder with tolerant wrapper
    _call_build_index(str(out_chunks), str(out_dir), args.embed_model, int(args.batch))

    try:
        print_index_stats(str(out_dir))
    except Exception:
        print(f"[rebuild] Index saved to {out_dir} (stats printer not available)")

    print(f"[OK] Rebuild complete. Index at: {out_dir}")


# ----------------- subcommand: ask -----------------
def _fmt_sources(results: List[Dict[str, Any]]) -> str:
    lines = []
    seen = set()
    for r in results:
        md = r.get("metadata", {})
        src = md.get("source", "unknown")
        ci = md.get("chunk_index", "?")
        score = r.get("score", None)
        key = (src, ci)
        if key in seen:
            continue
        seen.add(key)
        if score is None:
            lines.append(f"- {src}#chunk{ci}")
        else:
            lines.append(f"- {src}#chunk{ci}  (score={score:.4f})")
    return "\n".join(lines) if lines else "- (no sources)"


def cmd_ask(args: argparse.Namespace) -> None:
    index_dir: Path = Path(args.index)
    if not index_dir.exists():
        raise FileNotFoundError(f"[ask] index folder not found: {index_dir}. Run 'rebuild' first.")

    index, metas, manifest = load_index(str(index_dir))
    ntotal = getattr(index, "ntotal", 0)
    if ntotal == 0:
        raise RuntimeError("[ask] Empty FAISS index. Rebuild first.")

    q = args.query.strip()
    if not q:
        raise ValueError("[ask] Empty query provided.")
    qvec = embed_query(q, model=args.embed_model)

    results = retrieve(index, metas, qvec, k=args.k)
    if not results:
        print("No results found (k too small or index too tiny).")
        return

    thresh = args.score_threshold
    if thresh is not None:
        best = results[0].get("score", 0.0)
        if best < thresh:
            print("Not enough evidence in your notes to answer confidently.\n")
            print("Top sources:\n" + _fmt_sources(results[:2]))
            return

    context = format_context(results)
    user_msg = USER_PROMPT.format(question=q, context=context)
    answer = chat_complete(SYSTEM_PROMPT, user_msg, model=args.chat_model)

    print("\n=== ANSWER ===\n")
    print(answer.strip())
    print("\n=== SOURCES ===")
    print(_fmt_sources(results))


# ----------------- argparse wiring -----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Notes RAG CLI (rebuild | ask)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # defaults from env (overridable via flags)
    d_chunk_size = env_or("CHUNK_SIZE", 1000)
    d_chunk_overlap = env_or("CHUNK_OVERLAP", 200)
    d_embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    d_batch = env_or("BATCH_SIZE", 128)
    d_k = env_or("RETRIEVAL_K", 5)
    d_chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # rebuild
    sp = sub.add_parser("rebuild", help="Load -> chunk -> embed -> build FAISS index")
    sp.add_argument("--data", default="data", help="Folder with source files (md/pdf)")
    sp.add_argument("--chunks", default="index/chunks.jsonl", help="Output JSONL with chunks")
    sp.add_argument("--out", default="index", help="Output folder for FAISS + metadata")
    sp.add_argument("--size", type=int, default=d_chunk_size, help="Chunk size")
    sp.add_argument("--overlap", type=int, default=d_chunk_overlap, help="Chunk overlap")
    sp.add_argument("--embed_model", default=d_embed_model, help="Embedding model name")
    sp.add_argument("--batch", type=int, default=d_batch, help="Embedding batch size")
    sp.set_defaults(func=cmd_rebuild)

    # ask
    sa = sub.add_parser("ask", help="Retrieve -> LLM answer")
    sa.add_argument("--index", default="index", help="Folder with FAISS index + metadata")
    sa.add_argument("--query", required=True, help="Your question")
    sa.add_argument("--k", type=int, default=d_k, help="Top-k chunks to retrieve")
    sa.add_argument("--embed_model", default=d_embed_model, help="Embedding model name")
    sa.add_argument("--chat_model", default=d_chat_model, help="Chat model name")
    sa.add_argument("--score_threshold", type=float, default=None, help="Minimum top score to answer")
    sa.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()