# services/audiobook_manager.py (or src/audiobook_manager.py depending on your tree)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
import logging

from config import AUDIOBOOK_LANGUAGE, AUDIOBOOK_LANGUAGE_CODE
from src.audio_engine import AudioEngine
from src.utils.captioning_utils import ImageCaptioner
import streamlit as st
from src.utils.pdf_utils import (
    extract_page_images,
    extract_text_by_page,
    get_pdf_metadata,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, str], None]

"""
Handles voice + speed selection (language is fixed to AUDIOBOOK_LANGUAGE) and routes work
through the PDF/text utilities and AudioEngine.

Also:
- Renders each PDF page to an image
- Uses ImageCaptioner (best_epoch model) to describe pages
- Appends captions into the text before generating audio
"""


# -----------------------------------------------------------------------------


@dataclass
class AudiobookSettings:
    voice: Optional[str] = None
    speed: float = 1.0


# -----------------------------------------------------------------------------


class AudiobookManager:
    """
    High-level orchestrator for creating audiobooks from PDF files.

    With image captioning:
      - text per page via pdf_utils.extract_text_by_page
      - per-page images rendered & captioned via ImageCaptioner
      - captions appended into the script so TTS reads them aloud
    """

    def __init__(
        self,
        data_dir: Path | str = Path("data"),
        audio_engine: Optional[AudioEngine] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdfs"
        self.audio_dir = self.data_dir / "audio"
        self.text_dir = self.data_dir / "text"
        self.images_dir = self.data_dir / "page_images"

        for d in (self.pdf_dir, self.audio_dir, self.text_dir, self.images_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.audio_engine = audio_engine or AudioEngine(
            cache_dir=self.data_dir / "audio_cache"
        )

        # Captioning model (best_epoch)
# ---------------------------------------------------------
# Load Image Captioner using correct checkpoint path
# ---------------------------------------------------------
        try:
            # model is located at: src/training/best_epoch.pth
            project_root = Path(__file__).resolve().parents[1]   # -> src/
            ckpt_path = project_root / "training" / "best_epoch.pth"

            self.captioner = ImageCaptioner(checkpoint_path=ckpt_path)

            logger.info(f"ImageCaptioner loaded from {ckpt_path}")

        except Exception as exc:
            logger.error(f"Captioner error: {exc}")
            self.captioner = None


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
                logger.exception("Progress callback raised an exception")

        logger.info("[%3d%%] %s", percent, message)

    def _generate_page_captions(
        self,
        pdf_path: Path,
        book_id: str,
    ) -> Dict[int, str]:

        if self.captioner is None:
            logger.warning("ImageCaptioner not available; skipping image captions.")
            return {}

        # Each book gets its own subdirectory for page images
        out_dir = self.images_dir / book_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract page images
        page_images = extract_page_images(str(pdf_path), str(out_dir))
        if not page_images:
            logger.warning("No page images extracted.")
            return {}

        # 2. Convert dictionary → sorted list of paths
        pages_sorted = sorted(page_images.keys())
        image_paths = [Path(page_images[p]) for p in pages_sorted]

        # 3. Show Streamlit info AFTER image_paths exists
        st.info(f"Generating captions for {len(image_paths)} page images…")

        # 4. Generate captions
        captions = self.captioner.generate_batch(image_paths)
        page_captions: Dict[int, str] = {}

        for page_num, caption in zip(pages_sorted, captions):
            if caption and caption.strip():
                page_captions[page_num] = caption.strip()

        return page_captions


        

        return page_captions

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

        Steps:
            - Extract text per page
            - Generate image captions per page (if model available)
            - Append "[Image description: ...]" to the page text
            - Send the enriched script to AudioEngine
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
            self._emit_progress(progress_callback, 20, "Preparing PDF extraction")
            ocr_codes = [AUDIOBOOK_LANGUAGE_CODE]
            self._emit_progress(progress_callback, 35, "Extracting text from PDF")
            page_texts = extract_text_by_page(
                str(pdf_path),
                enable_ocr=True,
                ocr_languages=ocr_codes,
            )
            metadata = get_pdf_metadata(str(pdf_path))

            # -----------------------------------------------------------------
            # 2) Generate image captions per page (optional)
            # -----------------------------------------------------------------
            self._emit_progress(progress_callback, 45, "Generating image captions")
            page_captions = self._generate_page_captions(
                pdf_path=pdf_path,
                book_id=book_id,
            )

            # Build final script: text + [Image description: ...]
            ordered_page_numbers = sorted(page_texts.keys())
            ordered_pages: List[str] = []

            for page_num in ordered_page_numbers:
                text = page_texts[page_num].strip()
                caption = page_captions.get(page_num)

                if caption:
                    enriched = (
                        text
                        + "\n\n"
                        + f"[Image description: {caption}]"
                    )
                else:
                    enriched = text

                ordered_pages.append(enriched)

            full_text = "\n\n".join(ordered_pages)

            if not full_text.strip():
                raise RuntimeError("No text could be extracted from the PDF.")

            txt_path = self.text_dir / f"{book_id}.txt"
            txt_path.write_text(full_text, encoding="utf-8")
            self._emit_progress(progress_callback, 60, f"Saved text to {txt_path}")

            # -----------------------------------------------------------------
            # 3) Generate audio with selected voice + speed
            # -----------------------------------------------------------------
            self._emit_progress(
                progress_callback,
                75,
                (
                    "Generating audio "
                    f"(voice={settings.voice or 'default'}, "
                    f"speed={settings.speed:.2f}x)…"
                ),
            )

            audio_bytes = self.audio_engine.generate_audio(
                full_text,
                language=AUDIOBOOK_LANGUAGE_CODE,
                voice=settings.voice,
                speed=settings.speed,
            )

            if not audio_bytes:
                raise RuntimeError("Audio engine produced empty output")

            audio_path = self.audio_dir / f"{book_id}.wav"
            audio_path.write_bytes(audio_bytes)
            self._emit_progress(progress_callback, 90, f"Saved audio to {audio_path}")

            # -----------------------------------------------------------------
            # 4) Estimate duration (adjusted for speed)
            # -----------------------------------------------------------------
            duration_str = self.estimate_duration(
                text_length=len(full_text),
                speed=settings.speed,
            )

            self._emit_progress(
                progress_callback, 100, "Audiobook created successfully"
            )

            return {
                "success": True,
                "audio_path": str(audio_path),
                "text_path": str(txt_path),
                "metadata": metadata,
                "total_pages": metadata.get("total_pages", len(page_texts)),
                "duration": duration_str,
                "language": AUDIOBOOK_LANGUAGE,
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
        # Rough estimate: 5 characters per word
        words = text_length / 5.0
        effective_wpm = max(1.0, wpm * max(0.1, speed))
        minutes = words / effective_wpm
        seconds = int(minutes * 60)

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
