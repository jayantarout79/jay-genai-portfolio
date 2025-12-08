import json
import os
from datetime import datetime
from typing import Dict, Any

from services import data_loader

WORD_COUNTS_PATH = os.path.join("data", "word_counts.json")


def compute_early_insights(raw_folder_path: str) -> Dict[str, Any]:
    user = data_loader.load_user_info(raw_folder_path)
    conversations = data_loader.load_conversations(raw_folder_path)
    messages_flat = data_loader.flatten_messages(conversations, author_filter="user")
    message_stats = data_loader.message_stats(messages_flat)
    monthly_counts = data_loader.message_monthly_counts(messages_flat)
    word_counts = data_loader.user_word_counts(messages_flat)
    longest_words = sorted(
        (
            {**wc, "length": len(wc["word"])}
            for wc in word_counts
            if wc.get("word")
        ),
        key=lambda x: (x["length"], x["count"]),
        reverse=True,
    )

    sessions, audio_files, _ = data_loader.scan_audio_sessions(raw_folder_path)
    images = data_loader.scan_images(raw_folder_path)
    audio_seconds = data_loader.sum_audio_durations_seconds(raw_folder_path)
    audio_minutes = round(audio_seconds / 60, 1) if audio_seconds else 0.0

    # Persist full word counts to a file for reuse / inspection.
    os.makedirs(os.path.dirname(WORD_COUNTS_PATH), exist_ok=True)
    try:
        with open(WORD_COUNTS_PATH, "w", encoding="utf-8") as f:
            json.dump(word_counts, f, indent=2)
    except OSError:
        # Non-fatal; continue without blocking UI.
        pass

    return {
        "user": user,
        "messages": {
            **message_stats,
            "by_month": monthly_counts,
            "top_words": word_counts[:10],
            "longest_words": longest_words[:10],
            "word_counts_path": WORD_COUNTS_PATH,
            "distinct_words": len(word_counts),
        },
        "audio": {"sessions": sessions, "files": audio_files, "minutes": audio_minutes},
        "images": images,
        "calculated_at": datetime.utcnow().isoformat() + "Z",
    }
