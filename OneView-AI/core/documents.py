"""Utilities for ingesting non-video uploads (CSV, PDF, TXT, images)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


def classify_kind(path: Path) -> str:
    """Return a normalized file kind based on extension."""
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v"}:
        return "video"
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".pdf"}:
        return "pdf"
    if suffix in {".txt", ".log"}:
        return "log"
    if suffix in {".jpg", ".jpeg", ".png"}:
        return "image"
    return "document"


def read_text_file(path: Path, limit: int = 6000) -> str:
    """Read plain text/log files with a safety limit."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""
    if len(content) > limit:
        return content[:limit] + "\n...[truncated]..."
    return content


def summarize_csv(path: Path, preview_rows: int = 8) -> str:
    """Produce a compact text summary of a CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - runtime guard
        return f"Failed to parse CSV: {exc}"
    head_preview = df.head(preview_rows).to_markdown(index=False)
    stats = df.describe(include="all").fillna("").head(6)
    stats_preview = stats.to_markdown()
    return (
        f"CSV file: {path.name}\n"
        f"Rows: {len(df)}, Columns: {', '.join(df.columns[:12])}\n"
        f"Sample rows:\n{head_preview}\n\n"
        f"Quick stats:\n{stats_preview}"
    )


def extract_pdf(path: Path, limit_pages: int = 12) -> str:
    """Extract text from a PDF if the dependency is installed."""
    if PdfReader is None:
        return "PDF parsing unavailable (install pypdf)."
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - runtime guard
        return f"Failed to read PDF: {exc}"
    text_parts: List[str] = []
    for idx, page in enumerate(reader.pages[:limit_pages]):
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(text_parts).strip()
    if len(text) > 8000:
        text = text[:8000] + "\n...[truncated]..."
    if not text:
        text = "No extractable text detected."
    return f"PDF file: {path.name}\n{text}"


def describe_image(path: Path) -> str:
    """Return lightweight metadata for image files."""
    if Image is None:
        return f"Image: {path.name} (install Pillow for dimensions)"
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
    except Exception:
        return f"Image: {path.name} (could not open)"
    return f"Image: {path.name} — {width}x{height}px, mode {mode}"


def ingest_files(files: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Given persisted uploads, return normalized descriptors and textual content blobs.

    Returns:
        (descriptors, text_blobs)
    """
    descriptors: List[Dict[str, str]] = []
    text_blobs: List[str] = []
    for item in files:
        path = Path(item["path"])
        kind = item.get("kind") or classify_kind(path)
        descriptor = {"name": item.get("name", path.name), "kind": kind, "path": str(path)}
        descriptors.append(descriptor)

        if kind == "csv":
            text_blobs.append(summarize_csv(path))
        elif kind == "pdf":
            text_blobs.append(extract_pdf(path))
        elif kind == "log" or kind == "document":
            text_blobs.append(read_text_file(path))
        elif kind == "image":
            text_blobs.append(describe_image(path))
        # videos are handled separately for transcription
    return descriptors, text_blobs
