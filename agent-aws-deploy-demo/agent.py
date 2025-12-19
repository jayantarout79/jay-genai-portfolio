from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class ResearchResult:
    topic: str
    generated_at: str
    brief: str
    sources: List[Dict[str, Any]]


def _cache_key(topic: str) -> str:
    safe = "".join(c for c in topic.lower().strip() if c.isalnum() or c in ("-", "_", " "))
    safe = safe.replace(" ", "_")[:80]
    return f"{safe}.json"


def load_from_cache(topic: str) -> ResearchResult | None:
    path = CACHE_DIR / _cache_key(topic)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResearchResult(**data)


def save_to_cache(result: ResearchResult) -> None:
    path = CACHE_DIR / _cache_key(result.topic)
    path.write_text(json.dumps(result.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")


def research_web(topic: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Fetch web sources using Tavily (fast + clean citations)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in .env")

    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": topic})
    # results is typically a list of dicts: {title, url, content}
    return results


def write_brief(topic: str, sources: List[Dict[str, Any]]) -> str:
    """Use OpenAI to turn sources into a ~500 word research brief with citations."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY in .env")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)

    # Build a compact context block
    source_block_lines = []
    for i, s in enumerate(sources, start=1):
        title = (s.get("title") or "Untitled").strip()
        url = (s.get("url") or "").strip()
        content = (s.get("content") or "").strip()
        source_block_lines.append(f"[{i}] {title}\nURL: {url}\nNotes: {content}\n")

    source_block = "\n".join(source_block_lines)

    prompt = f"""
You are a senior research analyst.

Task:
Write a concise research brief (~500 words) on: "{topic}"

Rules:
- Use only the SOURCES provided below.
- Include 3–6 short inline citations like [1], [2] when you make claims.
- End with a section: "Key Takeaways" (3–5 bullets).
- Tone: clear, neutral, executive-friendly (no hype).

SOURCES:
{source_block}
""".strip()

    resp = llm.invoke(prompt)
    return resp.content


def run_research_agent(topic: str, use_cache: bool = True, max_results: int = 5) -> ResearchResult:
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")

    if use_cache:
        cached = load_from_cache(topic)
        if cached:
            return cached

    sources = research_web(topic, max_results=max_results)
    brief = write_brief(topic, sources)

    result = ResearchResult(
        topic=topic,
        generated_at=datetime.utcnow().isoformat() + "Z",
        brief=brief,
        sources=sources,
    )
    save_to_cache(result)
    return result