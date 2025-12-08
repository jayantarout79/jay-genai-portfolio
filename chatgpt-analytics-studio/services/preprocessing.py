import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from services.data_loader import flatten_messages, load_conversations, scan_audio_sessions


def _ensure_clean_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    # Remove stale files to avoid mixing old/new outputs.
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if os.path.isfile(full_path):
            os.remove(full_path)


def chunk_conversations(raw_folder_path: str, chunk_dir: str, chunk_size: int = 200) -> Tuple[int, List[str]]:
    conversations = load_conversations(raw_folder_path)
    # Only keep user-authored prompts with non-empty text.
    messages = flatten_messages(conversations, author_filter="user")
    _ensure_clean_dir(chunk_dir)

    if not messages:
        return 0, []

    chunk_paths: List[str] = []
    for idx in range(0, len(messages), chunk_size):
        chunk = messages[idx : idx + chunk_size]
        chunk_path = os.path.join(chunk_dir, f"chunk_{len(chunk_paths)+1}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2)
        chunk_paths.append(chunk_path)
    return len(chunk_paths), chunk_paths


def generate_audio_transcripts(raw_folder_path: str, transcripts_dir: str) -> Tuple[int, List[str]]:
    sessions_count, files_total, sessions = scan_audio_sessions(raw_folder_path)
    _ensure_clean_dir(transcripts_dir)
    if sessions_count == 0:
        return 0, []

    transcript_paths: List[str] = []
    for session_path in sessions:
        session_name = os.path.basename(session_path)
        audio_files = [
            f for f in os.listdir(session_path) if os.path.splitext(f)[1].lower() in {".m4a", ".mp3", ".wav", ".ogg"}
        ]
        dummy_transcript = {
            "session": session_name,
            "files": audio_files,
            "transcript": f"Placeholder transcript for session {session_name} containing {len(audio_files)} files.",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        out_path = os.path.join(transcripts_dir, f"{session_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dummy_transcript, f, indent=2)
        transcript_paths.append(out_path)
    return len(transcript_paths), transcript_paths
