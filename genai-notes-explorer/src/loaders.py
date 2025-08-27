# src/loaders.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document

# Optional import; only needed if you have PDFs
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


ALLOWED_EXTS = {".md", ".txt", ".pdf"}
MAX_BYTES = 250 * 1024  # 250 KB


def load_directory(data_dir: str | Path) -> List[Dict]:
    """
    Recursively scan data_dir for .md/.txt/.pdf, skip hidden & large files,
    and return a flat list of document dicts:
    {
      "text": "...",
      "metadata": {
         "source": "data/filename.md",
         "type": "md|txt|pdf",
         # "page": 1   # only for PDFs
         # "title": "First heading"  # optional for markdown
      }
    }
    """
    root = Path(data_dir).resolve()
    docs: List[Dict] = []

    # Collect candidate files first (deterministic ordering)
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            # skip hidden dirs
            if path.name.startswith("."):
                continue
            continue

        # skip hidden files
        if path.name.startswith("."):
            continue

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        # size limit
        try:
            if path.stat().st_size > MAX_BYTES:
                print(f"[SKIP] {path} > {MAX_BYTES} bytes")
                continue
        except FileNotFoundError:
            continue

        candidates.append(path)

    # Sort by relative path for determinism
    candidates = sorted(candidates, key=lambda p: str(p.relative_to(root)))

    for file_path in candidates:
        ext = file_path.suffix.lower()
        rel_source = str(file_path.relative_to(root)).replace("\\", "/")
        if ext in {".md", ".txt"}:
            docs.extend(load_text_file(file_path, root_dir=root, source_override=rel_source))
        elif ext == ".pdf":
            docs.extend(load_pdf_file(file_path, root_dir=root, source_override=rel_source))
        else:
            # Shouldn't reach here due to ALLOWED_EXTS
            continue

    return docs


def load_text_file(path: str | Path, root_dir: str | Path, source_override: str | None = None) -> List[Dict]:
    """Load a .md/.txt file → 1 document with normalized text + metadata."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    text = _normalize_text(text)

    meta = {
        "source": source_override or _rel_source(p, root_dir),
        "type": p.suffix.lower().lstrip("."),
    }

    # Optional: capture first Markdown H1 as title
    if p.suffix.lower() == ".md":
        title = _first_markdown_h1(text)
        if title:
            meta["title"] = title

    return [{"text": text, "metadata": meta}]


def load_pdf_file(path: str | Path, root_dir: str | Path, source_override: str | None = None) -> List[Dict]:
    """Load a PDF → one document per page with normalized text + page metadata."""
    if PdfReader is None:
        raise ImportError(
            "pypdf is required to load PDFs. Install with: pip install pypdf"
        )

    p = Path(path)
    reader = PdfReader(str(p))
    docs: List[Dict] = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # Fix common hyphenation-at-linebreak pattern, then normalize
        raw = raw.replace("-\n", "")  # join split words
        raw = raw.replace("\u00ad\n", "")  # soft hyphen edge case
        text = _normalize_text(raw)

        meta = {
            "source": source_override or _rel_source(p, root_dir),
            "type": "pdf",
            "page": i,
        }
        docs.append({"text": text, "metadata": meta})

    return docs


# ---------------------- helpers ---------------------- #

def _rel_source(path: Path, root_dir: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root_dir).resolve())).replace("\\", "/")


def _normalize_text(s: str) -> str:
    """Normalize newlines, collapse excessive blank lines, strip outer whitespace."""
    # normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines → 2 newlines
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip()


def _first_markdown_h1(s: str) -> str | None:
    for line in s.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def load_documents(data_dir: str | Path):
    """
    Compatibility wrapper for chunker.py.
    Converts dict records returned by load_directory() into LangChain Document objects.
    """
    recs = load_directory(data_dir)
    return [Document(page_content=r["text"], metadata=r["metadata"]) for r in recs]


# ---------------------- quick manual test ---------------------- #
if __name__ == "__main__":
    import json
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = load_directory(data_dir)

    # Counts by type
    type_counts = {}
    for d in docs:
        t = d["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print("[INFO] Type counts:", type_counts)
    print("[INFO] Total docs:", len(docs))

    # Show first 2 metadata entries
    for d in docs[:2]:
        print("[META]", json.dumps(d["metadata"], ensure_ascii=False))

    # Show shortest/longest doc lengths
    if docs:
        lengths = [len(d["text"]) for d in docs]
        print("[INFO] Shortest len:", min(lengths), "Longest len:", max(lengths))

    # List pdf pages covered per file
    pdfs = {}
    for d in docs:
        md = d["metadata"]
        if md["type"] == "pdf":
            pdfs.setdefault(md["source"], []).append(md["page"])
    if pdfs:
        for src, pages in pdfs.items():
            print(f"[PDF] {src} pages: {sorted(pages)}")