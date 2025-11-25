"""Utilities for handling uploaded videos and extracting audio tracks."""

from __future__ import annotations

from pathlib import Path
import tempfile
import uuid
import logging
from typing import Optional, Tuple

try:
    from moviepy.editor import VideoFileClip
except ImportError:  # pragma: no cover - optional dependency check
    VideoFileClip = None  # type: ignore

try:
    import ffmpeg
except ImportError:  # pragma: no cover - optional dependency check
    ffmpeg = None  # type: ignore

logger = logging.getLogger(__name__)

TEMP_DIR = Path(tempfile.gettempdir()) / "multimodal_video_engine"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def persist_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded Streamlit file to disk for later processing."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(temp_path, "wb") as sink:
        sink.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded file to %s", temp_path)
    return temp_path


def _extract_with_moviepy(video_path: Path, audio_path: Path) -> float:
    global VideoFileClip
    if VideoFileClip is None:
        from moviepy.editor import VideoFileClip as _VideoFileClip  # type: ignore

        VideoFileClip = _VideoFileClip

    clip = VideoFileClip(str(video_path))
    duration = float(clip.duration or 0.0)
    try:
        audio = clip.audio
        if audio is None:
            raise ValueError("Uploaded video does not contain an audio track.")
        audio.write_audiofile(
            str(audio_path),
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            verbose=False,
            logger=None,
        )
    finally:
        clip.close()
    return duration


def _extract_with_ffmpeg(video_path: Path, audio_path: Path) -> float:
    if ffmpeg is None:
        raise RuntimeError(
            "moviepy or ffmpeg-python is required for audio extraction. Install dependencies via "
            "`pip install -r requirements.txt`."
        )
    try:
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe.get("format", {}).get("duration", 0.0) or 0.0)
    except ffmpeg.Error as exc:  # pragma: no cover - best effort
        logger.warning("ffprobe failed: %s", exc)
        duration = 0.0
    (
        ffmpeg.input(str(video_path))
        .output(
            str(audio_path),
            ac=1,
            ar=16000,
            format="wav",
            loglevel="error",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return duration


def extract_audio(video_path: Path, target_dir: Optional[Path] = None) -> Tuple[Path, float]:
    """
    Extract the audio track from a video file.

    Returns:
        Tuple containing the audio path and the video duration in seconds.
    """
    target_dir = target_dir or TEMP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    audio_path = target_dir / f"{video_path.stem}_{uuid.uuid4().hex}.wav"

    global VideoFileClip
    duration = 0.0
    extracted = False
    if VideoFileClip is not None:
        try:
            duration = _extract_with_moviepy(video_path, audio_path)
            extracted = True
        except ImportError:
            VideoFileClip = None  # type: ignore
        except Exception as exc:
            logger.warning("MoviePy extraction failed (%s). Falling back to ffmpeg.", exc)
            VideoFileClip = None  # type: ignore
            extracted = False

    if not extracted:
        duration = _extract_with_ffmpeg(video_path, audio_path)
        extracted = True

    logger.info("Extracted audio to %s", audio_path)
    return audio_path, duration
