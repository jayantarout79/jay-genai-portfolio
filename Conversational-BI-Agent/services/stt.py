# services/stt.py
import io
import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# --- Simple magic-byte detection to choose proper extension/MIME ---
_MAGIC_WAV = b"RIFF"               # WAV header
_MAGIC_WEBM = b"\x1a\x45\xdf\xa3"   # EBML header (WebM/Matroska)
_MAGIC_MP3 = b"\xff\xfb"           # common MPEG1 Layer3 sync

_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".webm": "audio/webm",
}


def _guess_ext(data: bytes) -> str:
    head = bytes(data[:16]) if data else b""
    if head.startswith(_MAGIC_WAV):
        return ".wav"
    if head.startswith(_MAGIC_WEBM):
        return ".webm"
    if head.startswith(_MAG3 := _MAGIC_MP3):  # alias to avoid linter warning
        return ".mp3"
    return ".wav"  # default


def _bytes_to_file_tuple(data: bytes) -> tuple:
    ext = _guess_ext(data)
    mime = _MIME_BY_EXT.get(ext, "application/octet-stream")
    name = f"speech{ext}"
    return (name, data, mime)


def transcribe_with_openai(audio_bytes: bytes) -> str:
    """Transcribe audio bytes via OpenAI's /v1/audio/transcriptions with model fallbacks.
    Avoids SDK to prevent accidental chat/responses routing that can cause `messages` errors.
    """
    if not isinstance(audio_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("transcribe_with_openai expects raw bytes")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY — please export OPENAI_API_KEY before running.")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": _bytes_to_file_tuple(bytes(audio_bytes))}

    # Try in order; if 4o-transcribe isn't enabled on the account, we fall back.
    models = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]

    last_err: Optional[str] = None
    for model in models:
        data = {"model": model, "response_format": "text"}
        try:
            r = requests.post(url, headers=headers, files=files, data=data, timeout=90)
            if r.status_code == 200 and r.text:
                return r.text.strip()
            # capture diagnostic and try next model
            last_err = f"HTTP {r.status_code}: {r.text.strip()} (model={model}, file={files['file'][0]})"
        except Exception as e:
            last_err = f"Exception: {e} (model={model}, file={files['file'][0]})"

    raise RuntimeError(f"OpenAI transcription failed. Last error: {last_err}")