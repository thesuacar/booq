from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import logging
from tempfile import NamedTemporaryFile

import pyttsx3  

logger = logging.getLogger(__name__)


class AudioEngine:
    """
    Offline TTS engine using pyttsx3.

    Generates WAV audio and supports simple caching based on text + settings.
    """

    def __init__(self, cache_dir: Path | str = Path("data/audio_cache")) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialise engine once (sync & single-process use)
        self.engine = pyttsx3.init()

    # --------- Public API ---------

    def generate_audio(
        self,
        text: str,
        *,
        language: str = "en",
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Generate WAV audio for the given text.

        Args:
            text: Input text.
            language: Kept for API compatibility (pyttsx3 uses system voices).
            voice: Optional substring of desired voice name.
            speed: Multiplier for speaking rate (1.0 = default).

        Returns:
            WAV audio bytes.
        """
        if not text.strip():
            logger.warning("generate_audio called with empty text")
            return b""

        cache_key = self._generate_cache_key(text, language, voice, speed)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.info("Using cached audio for key %s", cache_key)
            return cached

        logger.info("Generating new audio (len=%d chars)", len(text))
        audio_bytes = self._synthesise(text, voice=voice, speed=speed)
        self._save_to_cache(cache_key, audio_bytes)
        return audio_bytes

    # --------- Internal helpers ---------

    def _synthesise(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Perform the actual TTS call using pyttsx3 and return WAV bytes.
        """
        # Configure engine
        if voice:
            self._set_voice(voice)

        self._set_speed(speed)

        # pyttsx3 only writes to file, so use a temporary WAV
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = Path(tmp.name)

        try:
            self.engine.save_to_file(text, str(temp_path))
            self.engine.runAndWait()

            with temp_path.open("rb") as f:
                audio_bytes = f.read()

            return audio_bytes
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def _set_voice(self, voice_name: str) -> None:
        """
        Try to select a voice whose name contains the given substring (case-insensitive).
        """
        try:
            voices = self.engine.getProperty("voices")
            for v in voices:
                if voice_name.lower() in (v.name or "").lower():
                    self.engine.setProperty("voice", v.id)
                    logger.info("Selected voice: %s", v.name)
                    return
            logger.warning("Requested voice '%s' not found; using default", voice_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Error while setting voice '%s': %s", voice_name, exc)

    def _set_speed(self, speed: float) -> None:
        """
        Adjust the speaking rate based on a multiplier.
        """
        try:
            base_rate = self.engine.getProperty("rate")
            # Clamp speed to a reasonable range
            speed = max(0.5, min(speed, 2.0))
            new_rate = int(base_rate * speed)
            self.engine.setProperty("rate", new_rate)
            logger.info("Set speech rate to %d (base=%d, speed=%.2f)", new_rate, base_rate, speed)
        except Exception as exc:  # pragma: no cover
            logger.exception("Error while setting speed: %s", exc)

    # --------- Caching ---------

    def _generate_cache_key(
        self,
        text: str,
        language: str,
        voice: Optional[str],
        speed: float,
    ) -> str:
        payload = f"{language}|{voice}|{speed}|{text}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _get_from_cache(self, key: str) -> Optional[bytes]:
        cache_file = self.cache_dir / f"{key}.wav"
        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    return f.read()
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed to read cache file %s: %s", cache_file, exc)
        return None

    def _save_to_cache(self, key: str, audio_bytes: bytes) -> None:
        cache_file = self.cache_dir / f"{key}.wav"
        try:
            with cache_file.open("wb") as f:
                f.write(audio_bytes)
            logger.info("Saved audio to cache: %s", cache_file)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to save cache file %s: %s", cache_file, exc)
