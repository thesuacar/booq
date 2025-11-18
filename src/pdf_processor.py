from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import logging

import PyPDF2

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF text extraction, metadata retrieval and saving merged text to disk.
    """

    def __init__(self, text_dir: Path | str = Path("data/text")) -> None:
        self.text_dir = Path(text_dir)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str | Path) -> Dict[int, str]:
        """
        Extract text from a PDF file page by page.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict mapping 1-based page numbers to text content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        page_texts: Dict[int, str] = {}

        try:
            with pdf_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                logger.info("Extracting text from %s (%d pages)", pdf_path, total_pages)

                for idx, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    # Normalise line endings
                    page_texts[idx] = text.replace("\r\n", "\n").replace("\r", "\n")

            logger.info("Successfully extracted text from %d pages", len(page_texts))
        except Exception as exc:  # pragma: no cover - safety log
            logger.exception("Error while extracting text from PDF %s: %s", pdf_path, exc)
            raise

        return page_texts

    def get_pdf_metadata(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Extract simple metadata from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict containing title, author, subject and total_pages.
        """
        pdf_path = Path(pdf_path)
        metadata: Dict[str, Any] = {}
        try:
            with pdf_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                raw_meta = reader.metadata or {}
                metadata = {
                    "total_pages": len(reader.pages),
                    "title": raw_meta.get("/Title", "") if raw_meta else "",
                    "author": raw_meta.get("/Author", "") if raw_meta else "",
                    "subject": raw_meta.get("/Subject", "") if raw_meta else "",
                }
        except Exception as exc:  # pragma: no cover - safety log
            logger.exception("Error while reading PDF metadata from %s: %s", pdf_path, exc)

        return metadata

    @staticmethod
    def combine_page_texts(page_texts: Dict[int, str]) -> str:
        """
        Combine page texts into a single string, preserving page order.

        Args:
            page_texts: Dict of page_number -> text.

        Returns:
            Single merged text string.
        """
        # Sort by page number to be safe
        ordered_pages = [page_texts[p] for p in sorted(page_texts.keys())]
        return "\n\n".join(ordered_pages)

    def save_full_text(self, full_text: str, book_id: str) -> Path:
        """
        Save merged text to a .txt file for a given book.

        Args:
            full_text: The complete text content of the book.
            book_id: Unique identifier for the book.

        Returns:
            Path to the saved .txt file.
        """
        txt_path = self.text_dir / f"{book_id}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with txt_path.open("w", encoding="utf-8") as f:
            f.write(full_text)

        logger.info("Saved merged text for book '%s' to %s", book_id, txt_path)
        return txt_path

    def process_pdf(
        self, pdf_path: str | Path, book_id: str
    ) -> Tuple[Dict[int, str], str, Path, Dict[str, Any]]:
        """
        High-level helper: extract, combine, save.

        Args:
            pdf_path: Path to PDF.
            book_id: Book identifier.

        Returns:
            (page_texts, full_text, txt_path, metadata)
        """
        page_texts = self.extract_text_from_pdf(pdf_path)
        full_text = self.combine_page_texts(page_texts)
        txt_path = self.save_full_text(full_text, book_id)
        metadata = self.get_pdf_metadata(pdf_path)
        if "total_pages" not in metadata:
            metadata["total_pages"] = len(page_texts)

        return page_texts, full_text, txt_path, metadata
