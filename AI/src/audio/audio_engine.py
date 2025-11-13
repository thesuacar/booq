"""Utilities for generating audio narration from text captions."""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from gtts import gTTS

# --- Optional IPython import for notebook playback ---
try:
    from IPython.display import Audio, display  # type: ignore
except ImportError:
    Audio = None  # type: ignore
    display = None  # type: ignore


@dataclass(frozen=True)
class AudioClip:
    """Represents a generated audio clip for a single caption."""

    caption: str
    path: Path
    generation_time: float

    def play(self, autoplay: bool = True) -> None:
        """Play the clip inline in notebook environments if IPython is available."""
        if Audio is None or display is None:
            raise RuntimeError(
                "IPython is not installed; playback is unavailable outside notebook environments."
            )
        display(Audio(filename=str(self.path), autoplay=autoplay))


def synthesize_captions(
    captions: Sequence[str],
    *,
    output_dir: Optional[Path] = None,
    prefix: str = "caption_audio",
    lang: str = "en",
    limit: Optional[int] = None,
) -> List[AudioClip]:
    """Convert captions to audio files using gTTS.

    Args:
        captions: Ordered text captions to narrate.
        output_dir: Directory to store generated audio. Defaults to the current directory.
        prefix: File name prefix for generated audio files.
        lang: Language code passed to gTTS.
        limit: Optional cap on the number of captions to synthesize.

    Returns:
        A list of AudioClip metadata for the generated audio files.
    """
    output_path = Path(output_dir) if output_dir is not None else Path.cwd()
    output_path.mkdir(parents=True, exist_ok=True)

    clips: List[AudioClip] = []
    for index, caption in enumerate(captions):
        if limit is not None and index >= limit:
            break

        start = time.perf_counter()
        filename = f"{prefix}_{index + 1}.mp3"
        file_path = output_path / filename

        success = False
        for attempt in range(1, GTTS_MAX_ATTEMPTS + 1):
            try:
                tts = gTTS(text=caption, lang=lang)
                tts.save(str(file_path))
                success = True
                break
            except Exception as e:
                if attempt == GTTS_MAX_ATTEMPTS:
                    print(f"[ERROR] Failed to generate audio for caption {index + 1}: {e}")
                else:
                    delay = GTTS_RETRY_BASE_DELAY * attempt
                    print(f"[WARN] gTTS attempt {attempt} failed for caption {index + 1}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        if not success:
            continue

        elapsed = time.perf_counter() - start
        clips.append(AudioClip(caption=caption, path=file_path, generation_time=elapsed))

        if GTTS_THROTTLE_DELAY > 0:
            time.sleep(GTTS_THROTTLE_DELAY)

    return clips


def average_generation_time(clips: Iterable[AudioClip]) -> float:
    """Compute the mean generation time for a batch of clips."""
    clip_list = list(clips)
    if not clip_list:
        return 0.0
    return sum(clip.generation_time for clip in clip_list) / len(clip_list)


def export_generation_report(
    clips: Sequence[AudioClip],
    *,
    output_path: Path,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Persist audio generation metadata to disk as JSON and return the payload."""
    payload = {
        "clips": [
            {
                "caption": clip.caption,
                "file": str(clip.path),
                "generation_time": clip.generation_time,
            }
            for clip in clips
        ],
        "average_generation_time": average_generation_time(clips),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload
GTTS_MAX_ATTEMPTS = int(os.getenv("BOOQ_TTS_MAX_ATTEMPTS", "3"))
GTTS_RETRY_BASE_DELAY = float(os.getenv("BOOQ_TTS_RETRY_BASE_DELAY", "2.0"))
GTTS_THROTTLE_DELAY = float(os.getenv("BOOQ_TTS_THROTTLE_DELAY", "1.0"))
