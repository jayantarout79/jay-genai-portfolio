import json
import os
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

from openai import OpenAI

POSITIVE_WORDS = {"great", "good", "love", "happy", "helpful", "excellent", "win", "nice", "success"}
NEGATIVE_WORDS = {"bad", "sad", "angry", "hate", "issue", "problem", "bug", "fail", "error"}
STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "a",
    "i",
    "it",
    "in",
    "that",
    "is",
    "for",
    "you",
    "on",
    "with",
    "this",
    "was",
    "my",
    "have",
    "but",
}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_ENV_LOADED = False


def _get_openai_client() -> OpenAI:
    global _ENV_LOADED

    def _load_env_candidates():
        candidates = []
        cwd_env = Path(os.getcwd()) / ".env"
        candidates.append(cwd_env)
        here_env = Path(__file__).resolve().parent.parent / ".env"
        candidates.append(here_env)
        # Also walk up parents to root to find first .env (limit depth to avoid huge walks)
        for parent in list(Path(__file__).resolve().parents)[:6]:
            cand = parent / ".env"
            candidates.append(cand)
        for cand in candidates:
            if cand.is_file():
                try:
                    with cand.open("r", encoding="utf-8") as f:
                        for line in f:
                            if "=" in line and not line.strip().startswith("#"):
                                k, v = line.strip().split("=", 1)
                                os.environ.setdefault(k, v)
                    break
                except OSError:
                    continue

    if not _ENV_LOADED:
        _load_env_candidates()
        _ENV_LOADED = True

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def _load_text_from_json(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict):
                texts.append(item.get("text") or "")
        return "\n".join(texts)
    if isinstance(data, dict):
        if "transcript" in data:
            return data.get("transcript") or ""
        return json.dumps(data)
    return ""


def _load_unit(path: str) -> Dict:
    """
    Load a chunk/transcript file and describe its structure for downstream analysis.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"type": "text", "text": ""}
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return {"type": "messages", "messages": data}
    if isinstance(data, dict) and "transcript" in data:
        return {"type": "text", "text": data.get("transcript", "")}
    return {"type": "text", "text": json.dumps(data)}


def _extract_topics(text: str, top_k: int = 5) -> List[Tuple[str, int]]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(tokens)
    return counts.most_common(top_k)


def _sentiment_score(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _analyze_unit(path: str, label: str) -> Dict:
    text = _load_text_from_json(path)
    topics = _extract_topics(text)
    sentiment = _sentiment_score(text)
    patterns = []
    if any("code" in t[0] for t in topics):
        patterns.append("Coding heavy chunk")
    if any("travel" in t[0] for t in topics):
        patterns.append("Travel discussions present")
    if not patterns and topics:
        patterns.append(f"General focus on {topics[0][0]}")
    return {
        "label": label,
        "path": path,
        "topics": [{"topic": t, "count": c} for t, c in topics],
        "sentiment_score": sentiment,
        "patterns": patterns,
    }


def run_parallel_analysis(chunk_paths: List[str], transcript_paths: List[str]) -> Dict:
    analysis_items: List[Dict] = []
    topics_counter: Counter = Counter()
    sentiments: List[Tuple[str, float]] = []

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, path in enumerate(chunk_paths, start=1):
            tasks.append(executor.submit(_analyze_unit, path, f"Conversation Chunk {idx}"))
        for idx, path in enumerate(transcript_paths, start=1):
            tasks.append(executor.submit(_analyze_unit, path, f"Audio Session {idx}"))

    for task in tasks:
        result = task.result()
        analysis_items.append(result)
        topics_counter.update({t["topic"]: t["count"] for t in result["topics"]})
        sentiments.append((result["label"], result["sentiment_score"]))

    sentiment_series = [
        {"label": label, "score": score, "index": idx}
        for idx, (label, score) in enumerate(sentiments)
    ]
    top_topics = [{"topic": t, "count": c} for t, c in topics_counter.most_common(15)]
    patterns = [p for item in analysis_items for p in item.get("patterns", [])]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": analysis_items,
        "top_topics": top_topics,
        "sentiment_series": sentiment_series,
        "patterns": patterns,
        "models_used": ["heuristic"],
    }


def _chat_json(model: str, messages: List[Dict], temperature: float = 0.2) -> Dict:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}


def _analyze_chunk_openai(unit: Dict, label: str, model: str) -> Dict:
    # If we have structured user messages, analyze per message; else fall back to text summary.
    if unit.get("type") == "messages":
        msgs = unit.get("messages") or []
        # Keep only user prompts with >=7 words.
        prompts = []
        for idx, m in enumerate(msgs):
            text = (m.get("text") or "").strip()
            if not text:
                continue
            if len(text.split()) < 7:
                continue
            prompts.append({"id": idx + 1, "text": text[:500]})
        if not prompts:
            return {
                "label": label,
                "topics": [],
                "primary_topic": "",
                "sentiment_score": 0.0,
                "prompt_quality_score": None,
                "prompt_quality_label": "",
                "per_message": [],
                "summary": "",
            }
        payload = json.dumps(prompts)[:12000]
        prompt = (
            "Analyze each user prompt (array of {id,text}). For prompts with >=7 words, return per_message list of "
            "{id, topic_category (one of: technology, software engineering, data/analytics, research/writing, "
            "business/productivity, education, health/fitness, finance, creative, personal/other), "
            "prompt_quality_score (1-5), prompt_quality_label (Beginner/Intermediate/Advanced/Power), "
            "sentiment (-1..1)}. Also return summary (string), primary_topic (one category), "
            "topics (list of {category,note}), sentiment (overall average).\n"
            "Respond ONLY in JSON."
        )
        data = _chat_json(
            model=model,
            messages=[
                {"role": "system", "content": "Be concise. Respond only with JSON."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
        )
        return {
            "label": label,
            "summary": data.get("summary", ""),
            "topics": data.get("topics", []) or [],
            "primary_topic": data.get("primary_topic", "") or "",
            "sentiment_score": data.get("sentiment", 0.0) or 0.0,
            "prompt_quality_score": data.get("prompt_quality_score"),
            "prompt_quality_label": data.get("prompt_quality_label", ""),
            "per_message": data.get("per_message", []) or [],
        }

    # Fallback: treat as plain text chunk.
    text = unit.get("text", "") or ""
    truncated = text[:12000] if text else ""
    if not truncated:
        return {
            "label": label,
            "topics": [],
            "sentiment_score": 0.0,
            "summary": "",
            "prompt_quality_score": None,
            "prompt_quality_label": "",
            "primary_topic": "",
        }
    prompt = (
        "You are analyzing a conversation slice. Extract high-level topical category and prompt quality.\n"
        "Return JSON with keys:\n"
        "  summary (string),\n"
        "  topics (list of {category: one of ['technology','software engineering','data/analytics','research/writing','business/productivity','education','health/fitness','finance','creative','personal/other'], note}),\n"
        "  primary_topic (one of the same categories),\n"
        "  sentiment (number -1..1),\n"
        "  prompt_quality_score (1-5),\n"
        "  prompt_quality_label (Beginner/Intermediate/Advanced/Power).\n"
    )
    data = _chat_json(
        model=model,
        messages=[
            {"role": "system", "content": "Be concise. Respond only with JSON."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": truncated},
        ],
        temperature=0.2,
    )
    return {
        "label": label,
        "summary": data.get("summary", ""),
        "topics": data.get("topics", []) or [],
        "primary_topic": data.get("primary_topic", "") or "",
        "sentiment_score": data.get("sentiment", 0.0) or 0.0,
        "prompt_quality_score": data.get("prompt_quality_score"),
        "prompt_quality_label": data.get("prompt_quality_label", ""),
    }


def _final_openai_summary(items: List[Dict], model: str) -> Dict:
    data = _chat_json(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide concise, actionable analytics. Respond only in JSON with keys: "
                    "overall_summary (string), highlights (list), risks (list), recommendations (list), "
                    "user_profile (string), prompting_level (string: Beginner/Intermediate/Advanced), "
                    "tone_style (string), intent_patterns (list), improvement_tips (list)."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given these chunk analyses, produce overall insights and profile the user as a prompt writer.\n"
                    f"{json.dumps(items)[:12000]}"
                ),
            },
        ],
        temperature=0.3,
    )
    return {
        "overall_summary": data.get("overall_summary", ""),
        "highlights": data.get("highlights", []),
        "risks": data.get("risks", []),
        "recommendations": data.get("recommendations", []),
        "user_profile": data.get("user_profile", ""),
        "prompting_level": data.get("prompting_level", ""),
        "tone_style": data.get("tone_style", ""),
        "intent_patterns": data.get("intent_patterns", []),
        "improvement_tips": data.get("improvement_tips", []),
    }


def run_openai_analysis(chunk_paths: List[str], transcript_paths: List[str]) -> Dict:
    tasks = []
    analysis_items: List[Dict] = []
    topics_counter: Counter = Counter()
    sentiments: List[Tuple[str, float]] = []
    model = OPENAI_MODEL

    with ThreadPoolExecutor(max_workers=4) as executor:
        for idx, path in enumerate(chunk_paths, start=1):
            tasks.append(
                executor.submit(
                    _analyze_chunk_openai, _load_unit(path), f"Conversation Chunk {idx}", model
                )
            )
        for idx, path in enumerate(transcript_paths, start=1):
            tasks.append(
                executor.submit(
                    _analyze_chunk_openai, _load_unit(path), f"Audio Session {idx}", model
                )
            )

        for task in as_completed(tasks):
            result = task.result()
            analysis_items.append(result)
            primary = result.get("primary_topic")
            if primary:
                topics_counter[primary] += 1
            for t in result.get("topics", []):
                if isinstance(t, dict):
                    topic = t.get("category") or t.get("topic") or t.get("label")
                    if topic:
                        topics_counter[topic] += 1
            for pm in result.get("per_message", []):
                topic = pm.get("topic_category") or pm.get("topic")
                if topic:
                    topics_counter[topic] += 1
            sentiments.append((result.get("label", ""), result.get("sentiment_score", 0.0) or 0.0))

    sentiment_series = [
        {"label": label or f"Item {idx}", "score": score, "index": idx}
        for idx, (label, score) in enumerate(sentiments)
    ]
    top_topics = [{"topic": t, "count": c} for t, c in topics_counter.most_common(20)]
    # Aggregate prompt quality
    pq_scores = [it.get("prompt_quality_score") for it in analysis_items if isinstance(it.get("prompt_quality_score"), (int, float))]
    prompt_quality_avg = round(sum(pq_scores) / len(pq_scores), 2) if pq_scores else None
    final_insights = _final_openai_summary(analysis_items, model)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": analysis_items,
        "top_topics": top_topics,
        "sentiment_series": sentiment_series,
        "final_insights": final_insights,
        "prompt_quality_avg": prompt_quality_avg,
        "models_used": [model],
    }


def save_ai_aggregates(aggregates: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)
