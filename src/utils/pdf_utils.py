"""Utility helpers for extracting text, images, and metadata from PDFs."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import PyPDF2

try:  # Optional dependency for OCR fallback
    import easyocr
except Exception:  # pragma: no cover - optional install
    easyocr = None  # type: ignore

logger = logging.getLogger(__name__)


def extract_txt_from_pdf(
    pdf_path: str | Path,
    *,
    enable_ocr: bool = False,
    ocr_languages: Optional[List[str]] = None,
) -> str:
    """
    Extract textual content from the PDF and return a single UTF-8 string.

    Args:
        pdf_path: Input PDF file path.
        enable_ocr: When True, falls back to EasyOCR if a page lacks embedded text.
        ocr_languages: Optional EasyOCR language codes (e.g., ["en"]). Defaults to ["en"].
    """
    pages = extract_text_by_page(
        pdf_path,
        enable_ocr=enable_ocr,
        ocr_languages=ocr_languages,
    )
    ordered = [pages[idx] for idx in sorted(pages.keys()) if pages[idx]]
    return "\n\n".join(ordered).strip()


def extract_text_by_page(
    pdf_path: str | Path,
    *,
    enable_ocr: bool = False,
    ocr_languages: Optional[List[str]] = None,
) -> Dict[int, str]:
    """
    Return a mapping of page_number -> extracted text.
    If enable_ocr is True, EasyOCR is used as a fallback when PyMuPDF gives no text.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    lang_tuple = _normalise_languages(ocr_languages)
    doc = fitz.open(pdf_file)
    texts: Dict[int, str] = {}

    for page_index in range(len(doc)):
        page_num = page_index + 1
        page = doc.load_page(page_index)
        content = page.get_text("text").strip()

        if not content and enable_ocr:
            content = _extract_text_via_ocr(page, lang_tuple)

        texts[page_num] = content.strip()

    return texts


def extract_images_from_pdf(pdf_path: str | Path, output_dir: str | Path) -> List[str]:
    """
    Render every page of the PDF to an image file.
    Returns list of file paths ordered by page number.
    """
    page_images = extract_page_images(pdf_path, output_dir)
    return [page_images[p] for p in sorted(page_images.keys())]


def extract_page_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 200,
) -> Dict[int, str]:
    """Render each page to an image and return page_number -> image_path mapping."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_file)
    saved: Dict[int, str] = {}

    for index in range(len(doc)):
        page_num = index + 1
        page = doc.load_page(index)
        pix = page.get_pixmap(dpi=dpi)
        img_path = out_dir / f"page_{page_num:04d}.png"
        pix.save(str(img_path))
        saved[page_num] = str(img_path)

    return saved


def render_pages_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 200,
) -> List[str]:
    """
    Backwards-compatible helper used by ai_service.
    Simply renders each page to an image and returns an ordered list of paths.
    """
    return extract_images_from_pdf(pdf_path, output_dir)


def get_pdf_metadata(pdf_path: str | Path) -> Dict[str, str | int]:
    """Lightweight metadata helper mirroring PDFProcessor.get_pdf_metadata."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    try:
        with open(pdf_file, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            meta = reader.metadata or {}
            return {
                "total_pages": len(reader.pages),
                "title": meta.get("/Title", "") if hasattr(meta, "get") else "",
                "author": meta.get("/Author", "") if hasattr(meta, "get") else "",
            }
    except Exception as exc:  # pragma: no cover - PyPDF2 edge cases
        logger.warning("Failed to read metadata for %s: %s", pdf_file, exc)
        return {"total_pages": 0, "title": "", "author": ""}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_languages(languages: Optional[List[str]]) -> Tuple[str, ...]:
    if not languages:
        return ("en",)
    cleaned = tuple(sorted({lang.strip().lower() for lang in languages if lang}))
    return cleaned or ("en",)


@lru_cache(maxsize=8)
def _get_ocr_reader(languages: Tuple[str, ...]):
    if easyocr is None:
        logger.warning("easyocr not installed; OCR fallback disabled.")
        return None
    try:
        return easyocr.Reader(list(languages), gpu=False)
    except Exception as exc:
        logger.warning("Failed to initialise EasyOCR for %s: %s", languages, exc)
        return None


def _extract_text_via_ocr(page: fitz.Page, languages: Tuple[str, ...]) -> str:
    reader = _get_ocr_reader(languages)
    if reader is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        results = reader.readtext(img_bytes, detail=0)
        if not results:
            return ""
        return "\n".join(results).strip()
    except Exception as exc:
        logger.warning("EasyOCR failed on page: %s", exc)
        return ""
