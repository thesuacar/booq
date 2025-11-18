from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
import logging

from src.pdf_processor import PDFProcessor
from src.audio_engine import AudioEngine

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]


@dataclass
class AudiobookSettings:
    language: str = "en"      # for future use (neural TTS, multilingual)
    voice: Optional[str] = None
    speed: float = 1.0


class AudiobookManager:
    """
    High-level orchestrator for creating audiobooks from PDF files.

    This is used by the orchestrator_server (distributed architecture) and
    *not* directly by the UI.
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

        self.pdf_processor = pdf_processor or PDFProcessor(text_dir=self.text_dir)
        self.audio_engine = audio_engine or AudioEngine(cache_dir=self.data_dir / "audio_cache")

    def create_audiobook(
        self,
        pdf_path: str | Path,
        book_id: str,
        settings: Optional[AudiobookSettings] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete audiobook from a PDF.

        Returns a dict:
            success: bool
            audio_path: str
            text_path: str
            metadata: dict
            total_pages: int
            duration: str (HH:MM:SS)
            error: str (if failure)
        """
        settings = settings or AudiobookSettings()

        def update_progress(p: int, msg: str) -> None:
            if progress_callback:
                progress_callback(p, msg)
            logger.info("[%3d%%] %s", p, msg)

        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            update_progress(5, "Starting audiobook generation")

            # 1) Extract + save text
            update_progress(15, "Extracting text from PDF")
            page_texts, full_text, txt_path, metadata = self.pdf_processor.process_pdf(
                pdf_path, book_id
            )

            update_progress(35, "Generating audio from text")
            audio_bytes = self.audio_engine.generate_audio(
                full_text,
                language=settings.language,
                voice=settings.voice,
                speed=settings.speed,
            )

            if not audio_bytes:
                raise RuntimeError("Audio engine produced empty output")

            # 2) Save WAV file
            audio_path = self.audio_dir / f"{book_id}.wav"
            with audio_path.open("wb") as f:
                f.write(audio_bytes)
            update_progress(85, f"Saved audio to {audio_path}")

            # 3) Estimate duration
            duration_str = self._estimate_duration(len(full_text))

            update_progress(100, "Audiobook created successfully")

            return {
                "success": True,
                "audio_path": str(audio_path),
                "text_path": str(txt_path),
                "metadata": metadata,
                "total_pages": metadata.get("total_pages", len(page_texts)),
                "duration": duration_str,
            }

        except Exception as exc:
            logger.exception("Error while creating audiobook for book_id=%s: %s", book_id, exc)
            return {
                "success": False,
                "error": str(exc),
            }

    @staticmethod
    def _estimate_duration(text_length: int, wpm: int = 150) -> str:
        """Roughly estimate audio duration based on text length."""
        words = text_length / 5.0
        minutes = words / wpm
        seconds_total = int(minutes * 60)

        hours = seconds_total // 3600
        minutes = (seconds_total % 3600) // 60
        seconds = seconds_total % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

