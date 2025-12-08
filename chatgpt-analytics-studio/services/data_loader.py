import json
import os
import re
from typing import Any, Dict, List, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg"}


def load_user_info(raw_folder_path: str) -> Dict[str, Any]:
    user_path = os.path.join(raw_folder_path, "user.json")
    if not os.path.isfile(user_path):
        return {"name": "N/A", "email": "N/A", "dob": "N/A"}
    try:
        with open(user_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"name": "N/A", "email": "N/A", "dob": "N/A"}
    birth_year = data.get("birth_year") or data.get("dob") or "N/A"
    subscription = "ChatGPT Plus" if data.get("chatgpt_plus_user") else "Free / Unknown"
    return {
        "name": data.get("name") or "N/A",
        "email": data.get("email") or "N/A",
        "dob": birth_year,
        "subscription": subscription,
    }


def load_conversations(raw_folder_path: str) -> List[Dict[str, Any]]:
    conv_path = os.path.join(raw_folder_path, "conversations.json")
    if not os.path.isfile(conv_path):
        return []
    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def flatten_messages(conversations: List[Dict[str, Any]], author_filter: str = None) -> List[Dict[str, Any]]:
    """
    Convert OpenAI export structure to a flat list of messages.
    Falls back gracefully if expected fields are absent.
    """
    messages: List[Dict[str, Any]] = []
    for conv in conversations:
        conv_id = conv.get("id") or conv.get("conversation_id")
        title = conv.get("title") or "Untitled"
        conv_created = conv.get("create_time")
        mapping = conv.get("mapping") or {}
        for node in mapping.values():
            msg = node.get("message") if isinstance(node, dict) else None
            if not msg:
                continue
            author = (msg.get("author") or {}).get("role") or "unknown"
            if author_filter and author != author_filter:
                continue
            content = msg.get("content") or {}
            parts = content.get("parts") or []
            text = "\n".join([p for p in parts if isinstance(p, str)])
            if not text.strip():
                continue
            created_at = msg.get("create_time") or conv_created
            messages.append(
                {
                    "conversation_id": conv_id,
                    "conversation_title": title,
                    "author": author,
                    "text": text,
                    "created_at": created_at,
                }
            )
    return messages


def message_stats(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    total_messages = len(messages)
    total_chars = sum(len(m.get("text") or "") for m in messages)
    return {"total_messages": total_messages, "approx_chars": total_chars}


def message_monthly_counts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return counts per UTC month using message timestamp; falls back to conversation timestamp when missing.
    Messages without any timestamp are skipped.
    """
    from collections import Counter
    from datetime import datetime, timezone

    counts = Counter()
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None
    for msg in messages:
        created_at = msg.get("created_at")
        if created_at is None:
            continue
        dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
        key = f"{dt.year:04d}-{dt.month:02d}"
        counts[key] += 1
    return [{"month": month, "count": counts[month]} for month in sorted(counts)]


def scan_audio_sessions(raw_folder_path: str) -> Tuple[int, int, List[str]]:
    """
    Identify audio session folders by presence of audio files inside (including nested under e.g. <conv>/audio/).
    Returns (sessions_count, files_count, session_paths)
    """
    sessions: List[str] = []
    files_total = 0
    if not os.path.isdir(raw_folder_path):
        return 0, 0, []

    for entry in os.listdir(raw_folder_path):
        conv_path = os.path.join(raw_folder_path, entry)
        if not os.path.isdir(conv_path):
            continue
        has_audio = False
        for root, _, files in os.walk(conv_path):
            audio_files = [f for f in files if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
            if audio_files:
                has_audio = True
                files_total += len(audio_files)
        if has_audio:
            sessions.append(conv_path)
    return len(sessions), files_total, sessions


def scan_images(raw_folder_path: str) -> Dict[str, int]:
    uploaded = 0
    generated = 0
    if not os.path.isdir(raw_folder_path):
        return {"uploaded": 0, "generated": 0}

    # Uploaded images are top-level files matching export patterns.
    for entry in os.listdir(raw_folder_path):
        full_path = os.path.join(raw_folder_path, entry)
        if os.path.isfile(full_path):
            lower = entry.lower()
            ext = os.path.splitext(lower)[1]
            if ext in IMAGE_EXTENSIONS and (
                lower.startswith("file_") or lower.startswith("file-")
            ):
                uploaded += 1

    # Generated images live under dalle-generations or user-<id> folders.
    for entry in os.listdir(raw_folder_path):
        full_path = os.path.join(raw_folder_path, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.startswith("dalle-generations") or re.match("user-[\\w-]+", entry):
            for _, _, files in os.walk(full_path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                        generated += 1
    return {"uploaded": uploaded, "generated": generated}


def user_word_counts(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Count word frequencies from user-authored messages only.
    Returns a list of {"word": str, "count": int} sorted by count desc.
    """
    from collections import Counter
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None

    counts = Counter()
    stop_words = {
        "the",
        "and",
        "a",
        "to",
        "of",
        "in",
        "for",
        "is",
        "i",
        "you",
        "it",
        "on",
        "that",
        "this",
        "with",
        "at",
        "as",
        "be",
        "are",
        "was",
        "were",
        "am",
        "an",
        "by",
        "or",
        "we",
        "my",
        "me",
        "so",
        "but",
        "if",
        "from",
        "about",
        "up",
        "out",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "your",
        "our",
        "their",
        "them",
        "they",
        "what",
        "when",
        "how",
        "why",
        "which",
        "who",
        "whom",
        "will",
        "not",
        "have",
        "now",
        "all",
        "want",
        "create",
        "ai",
        "jkr",
    }
    word_re = re.compile(r"[A-Za-z0-9']+")
    for msg in messages:
        if (msg.get("author") or "").lower() != "user":
            continue
        text = msg.get("text") or ""
        for word in word_re.findall(text.lower()):
            if not word.isalpha():  # drop tokens with digits or punctuation
                continue
            if not re.search(r"[aeiou]", word):  # require at least one vowel
                continue
            if len(word) <= 2:  # drop tiny tokens
                continue
            if word in stop_words:
                continue
            if zipf_frequency and zipf_frequency(word, "en") < 2.0:
                # Skip unlikely English words (Zipf scale; 2.0 ~ infrequent but real words)
                continue
            counts[word] += 1
    return [{"word": w, "count": c} for w, c in counts.most_common()]


def sum_audio_durations_seconds(raw_folder_path: str) -> float:
    """
    Sum audio durations across session folders, returning total seconds.
    If mutagen is missing or a file can't be read, that file is skipped.
    """
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        MutagenFile = None

    total_seconds = 0.0
    if not os.path.isdir(raw_folder_path):
        return 0.0

    for entry in os.listdir(raw_folder_path):
        conv_path = os.path.join(raw_folder_path, entry)
        if not os.path.isdir(conv_path):
            continue
        for root, _, files in os.walk(conv_path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in AUDIO_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                # Try mutagen first, then fall back to wave for wav files.
                if MutagenFile:
                    try:
                        audio = MutagenFile(fpath)
                        if audio and hasattr(audio, "info") and getattr(audio.info, "length", None):
                            total_seconds += float(audio.info.length)
                            continue
                    except Exception:
                        pass
                if fpath.lower().endswith(".wav"):
                    import wave

                    try:
                        with wave.open(fpath, "rb") as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate() or 1
                            total_seconds += frames / float(rate)
                    except Exception:
                        continue
    return total_seconds
