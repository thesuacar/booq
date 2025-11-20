"""
Audio Engine for TTS (pyttsx3) - WAV output
Supports: voice selection, speed adjustments, caching
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict
from tempfile import NamedTemporaryFile
import os

import pyttsx3

logger = logging.getLogger(__name__)


class AudioEngine:
    """
    Thin wrapper around pyttsx3 that:

      - Lists available voices
      - Applies selected voice
      - Applies speed (rate) multiplier
      - Caches generated WAV files on disk

    It does NOT know anything about:
      - PDFs
      - jobs
      - users
      - HTTP
    """

    def __init__(self, cache_dir: Path | str = Path("data/audio_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Single engine instance
        self.engine = pyttsx3.init()
        self._default_rate = self.engine.getProperty("rate")

        self.friendly_voice_map = {}

        self.available_voices: Dict[str, str] = {}
        self._load_voices()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _load_voices(self) -> None:
        try:
            system_voices = self.engine.getProperty("voices")

            # Map system voices to friendly names
            voice_map = {
                "Microsoft David Desktop - English (United States)": "Tom",
                "Microsoft Zira Desktop - English (United States)": "Ana",
                "Microsoft Hazel Desktop - English (Great Britain)": "Emily",
            }

            friendly_mapping = {}

            for v in system_voices:
                friendly_name = voice_map.get(v.name)
                if friendly_name:
                    friendly_mapping[friendly_name] = v.id

            # Store mapping in BOTH attributes
            self.available_voices = friendly_mapping
            self.friendly_voice_map = friendly_mapping   # <-- REQUIRED FIX

            logger.info("Loaded voices: %s", list(self.available_voices.keys()))

        except Exception as exc:
            logger.error("Failed to load system voices: %s", exc)



    def _select_voice(self, voice_name: Optional[str]) -> None:
        """Select voice via friendly name."""
        if not voice_name:
            return

        system_id = self.available_voices.get(voice_name)
        if system_id:
            try:
                self.engine.setProperty("voice", system_id)
                return
            except Exception as exc:
                logger.error("Failed to set voice '%s': %s", voice_name, exc)

        logger.warning("Requested voice '%s' not found. Using default.", voice_name)

    
    def _set_speed(self, speed: float) -> None:
        """
        Apply a speed multiplier to the engine's rate.

        speed: 1.0 = normal, 2.0 = faster, 0.5 = slower.
        """
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0

        # Clamp to a safe range
        speed = max(0.5, min(2.0, speed))
        new_rate = int(self._default_rate * speed)
        try:
            self.engine.setProperty("rate", new_rate)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to set TTS rate: %s", exc)

    @staticmethod
    def _make_cache_key(text: str, language: str, voice: Optional[str], speed: float) -> str:
        """
        Build a stable hash key from the input settings.
        """
        m = hashlib.sha256()
        payload = f"{language}|{voice or ''}|{speed}|{text}".encode("utf-8")
        m.update(payload)
        return m.hexdigest()

    def _load_from_cache(self, key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{key}.wav"
        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    return f.read()
            except Exception as exc:
                logger.error("Failed to read cache file: %s", exc)
        return None

    def _save_to_cache(self, key: str, audio_bytes: bytes) -> None:
        cache_file = self.cache_dir / f"{key}.wav"
        try:
            with cache_file.open("wb") as f:
                f.write(audio_bytes)
            logger.info("Saved audio to cache: %s", cache_file)
        except Exception as exc:
            logger.error("Failed to save cache file %s: %s", cache_file, exc)

    # -------------------------------------------------------------------------
    # PUBLIC: Generate audio with caching
    # -------------------------------------------------------------------------

    def generate_audio(
        self,
        text: str,
        *,
        language: str = "en",
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Generate WAV audio from text, using optional voice + speed,
        with on-disk caching.

        Returns raw WAV bytes.
        """
        if not text.strip():
            raise ValueError("Cannot generate audio from empty text")

        cache_key = self._make_cache_key(text, language, voice, speed)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            logger.info("Returning cached audio for key=%s", cache_key)
            return cached

        # Apply voice + speed to engine
        self._select_voice(voice)
        self._set_speed(speed)

        # Use a temporary wav file, then read back
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.engine.save_to_file(text, str(tmp_path))
            self.engine.runAndWait()

            audio_bytes = tmp_path.read_bytes()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        self._save_to_cache(cache_key, audio_bytes)
        return audio_bytes
