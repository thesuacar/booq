from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
import logging

from src.pdf_processor import PDFProcessor
from src.audio_engine import AudioEngine

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]

"""
Takes language, voice, speed and passes them cleanly down into PDFProcessor and AudioEngine.

Handles OCR language mapping internally (UI can just send "English", "Spanish", etc.).

Returns a consistent dict with success, audio_path, text_path, duration, total_pages, metadata, error?.

All the “PDF → text → audio → duration” logic is here."""


# -----------------------------------------------------------------------------
# Settings dataclass
# -----------------------------------------------------------------------------

@dataclass
class AudiobookSettings:
    """
    High-level settings for audiobook generation.

    language:
        Human-readable language label coming from the UI, e.g.
        "English", "Spanish", "en", "es", etc.
        This is mapped internally to OCR language codes.

    voice:
        Name of a pyttsx3 voice. Must match one of
        AudioEngine.available_voices.keys(), or None for default.

    speed:
        Playback speed multiplier. 1.0 is normal speed.
        Values are clamped to a safe range by AudioEngine.
    """
    language: str = "English"
    voice: Optional[str] = None
    speed: float = 1.0


# -----------------------------------------------------------------------------
# Language mapping helpers
# -----------------------------------------------------------------------------

# UI / human names → EasyOCR language codes
_OCR_LANG_MAP: Dict[str, str] = {
    "english": "en",
    "en": "en",

    "spanish": "es",
    "es": "es",

    "french": "fr",
    "fr": "fr",

    "german": "de",
    "de": "de",

    "italian": "it",
    "it": "it",

    "portuguese": "pt",
    "pt": "pt",

    "korean": "ko",
    "ko": "ko",

    "japanese": "ja",
    "ja": "ja",
}


def _language_to_ocr_codes(language: str) -> List[str]:
    """
    Map a UI language string (e.g. 'English', 'en') to a list of EasyOCR codes.

    Fallback is ['en'] so we always have something that works.
    """
    if not language:
        return ["en"]

    key = language.strip().lower()
    code = _OCR_LANG_MAP.get(key, "en")
    return [code]


# -----------------------------------------------------------------------------
# AudiobookManager
# -----------------------------------------------------------------------------

class AudiobookManager:
    """
    High-level orchestrator for creating audiobooks from PDF files.

    This class is used by the orchestrator server and is NOT tied to Streamlit.
    It coordinates:
      - PDF text extraction (with OCR fallback)
      - Text saving
      - Audio generation via AudioEngine (voice + speed)
      - Duration estimation

    It is synchronous and filesystem-based so it works both locally and
    inside FastAPI services.
    """

    def __init__(
        self,
        data_dir: Path | str = Path("data"),
        pdf_processor: Optional[PDFProcessor] = None,
        audio_engine: Optional[AudioEngine] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdfs"
        self.audio_dir = self.data_dir / "audio"
        self.text_dir = self.data_dir / "text"

        for d in (self.pdf_dir, self.audio_dir, self.text_dir):
            d.mkdir(parents=True, exist_ok=True)

        # These can be overridden/injected (e.g. for tests or by orchestrator)
        self._pdf_processor = pdf_processor
        self.audio_engine = audio_engine or AudioEngine(
            cache_dir=self.data_dir / "audio_cache"
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_pdf_processor_for_language(self, language: str) -> PDFProcessor:
        """
        Either reuse the injected PDFProcessor or create a new one configured
        for the given language.
        """
        if self._pdf_processor is not None:
            return self._pdf_processor

        ocr_codes = _language_to_ocr_codes(language)
        logger.info("Creating PDFProcessor with OCR languages: %s", ocr_codes)
        return PDFProcessor(languages=ocr_codes)

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[ProgressCallback],
        percent: int,
        message: str,
    ) -> None:
        if progress_callback:
            try:
                progress_callback(percent, message)
            except Exception:
                # User code should never be allowed to crash the pipeline
                logger.exception("Progress callback raised an exception")

        logger.info("[%3d%%] %s", percent, message)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def create_audiobook(
        self,
        pdf_path: str | Path,
        book_id: str,
        settings: Optional[AudiobookSettings] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete audiobook from a PDF.

        Returns a dict with at least:
            success: bool
            audio_path: str        (on success)
            text_path: str         (on success)
            metadata: dict         (PDF metadata)
            total_pages: int       (page count)
            duration: str          (HH:MM:SS estimated, adjusted for speed)
            error: str             (on failure)
        """
        settings = settings or AudiobookSettings()

        try:
            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            self._emit_progress(progress_callback, 5, "Starting audiobook generation")

            # -----------------------------------------------------------------
            # 1) Extract + save text
            # -----------------------------------------------------------------
            self._emit_progress(progress_callback, 20, "Configuring PDF processor")
            pdf_processor = self._get_pdf_processor_for_language(settings.language)

            self._emit_progress(progress_callback, 35, "Extracting text from PDF")
            page_texts = pdf_processor.extract_text_from_pdf(str(pdf_path))
            metadata = pdf_processor.get_pdf_metadata(str(pdf_path))

            # Deterministic page ordering
            ordered_pages = [page_texts[p] for p in sorted(page_texts.keys())]
            full_text = "\n\n".join(ordered_pages)

            if not full_text.strip():
                raise RuntimeError("No text could be extracted from the PDF.")

            txt_path = self.text_dir / f"{book_id}.txt"
            txt_path.write_text(full_text, encoding="utf-8")
            self._emit_progress(progress_callback, 55, f"Saved text to {txt_path}")

            # -----------------------------------------------------------------
            # 2) Generate audio with selected voice + speed
            # -----------------------------------------------------------------
            self._emit_progress(
                progress_callback,
                70,
                (
                    "Generating audio "
                    f"(voice={settings.voice or 'default'}, "
                    f"speed={settings.speed:.2f}x)…"
                ),
            )

            audio_bytes = self.audio_engine.generate_audio(
                full_text,
                language=settings.language,  # used for caching key
                voice=settings.voice,
                speed=settings.speed,
            )

            if not audio_bytes:
                raise RuntimeError("Audio engine produced empty output")

            audio_path = self.audio_dir / f"{book_id}.wav"
            audio_path.write_bytes(audio_bytes)
            self._emit_progress(progress_callback, 90, f"Saved audio to {audio_path}")

            # -----------------------------------------------------------------
            # 3) Estimate duration (adjusted for speed)
            # -----------------------------------------------------------------
            duration_str = self.estimate_duration(
                text_length=len(full_text),
                speed=settings.speed,
            )

            self._emit_progress(progress_callback, 100, "Audiobook created successfully")

            return {
                "success": True,
                "audio_path": str(audio_path),
                "text_path": str(txt_path),
                "metadata": metadata,
                "total_pages": metadata.get("total_pages", len(page_texts)),
                "duration": duration_str,
            }

        except Exception as exc:
            logger.exception(
                "Error while creating audiobook for book_id=%s: %s", book_id, exc
            )
            return {
                "success": False,
                "error": str(exc),
            }

    # -------------------------------------------------------------------------
    # Duration helper
    # -------------------------------------------------------------------------

    def estimate_duration(
        self,
        text_length: int,
        speed: float = 1.0,
        wpm: int = 150,
    ) -> str:
        """
        Estimate audio duration based on text length and playback speed.

        - text_length: number of characters in the script
        - speed: playback speed multiplier (1.0 = normal, 2.0 = twice as fast)
        - wpm: base words-per-minute at speed 1.0

        Returns a "HH:MM:SS" string.
        """
        # Rough estimate: 5 characters per word
        words = text_length / 5.0

        # Adjust effective WPM by speed (faster speed = shorter duration)
        effective_wpm = max(1.0, wpm * max(0.1, speed))
        minutes = words / effective_wpm
        seconds = int(minutes * 60)

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
