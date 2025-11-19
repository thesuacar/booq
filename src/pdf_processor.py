"""
PDF Processor with EasyOCR (No Tesseract, No Poppler)
Supports multilingual OCR and works on Windows.
"""

import PyPDF2
import fitz  # PyMuPDF
import easyocr
from PIL import Image
from pathlib import Path
from typing import Dict


class PDFProcessor:
    def __init__(self, languages=None):
        """
        Initialize EasyOCR with provided languages.
        languages -> list of strings like ["en"], ["en","es"], ["fr"], etc.
        """
        if languages is None:
            languages = ["en"]  # Default English
        
        print(f"[PDFProcessor] Loading EasyOCR languages: {languages}")
        self.ocr_reader = easyocr.Reader(languages, gpu=False)

    # ---------------------------------------------------------------------
    # Convert PyMuPDF page → PIL Image
    # ---------------------------------------------------------------------
    def _page_to_image(self, page):
        """
        Render a PyMuPDF page into a PIL image.
        """
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    # ---------------------------------------------------------------------
    # OCR Extraction (fallback)
    # ---------------------------------------------------------------------
    def _extract_text_via_ocr(self, page) -> str:
        """Extract text with OCR from a PDF page using EasyOCR."""
        try:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")   # PNG bytes for EasyOCR
            results = self.ocr_reader.readtext(img_bytes, detail=0)
            return "\n".join(results)
        except Exception as exc:
            print(f"[PDFProcessor] OCR failed: {exc}")
            return ""

    # ---------------------------------------------------------------------
    # Main public method
    # ---------------------------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text page-by-page.
        Embedded text → primary
        OCR fallback → secondary
        """
        pdf_path = Path(pdf_path)
        text_per_page: Dict[int, str] = {}

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[PDFProcessor] Extracting text from {total_pages} pages.")

        for i in range(total_pages):
            page_num = i + 1
            page = doc.load_page(i)

            # Try embedded text first
            extracted = page.get_text()
            if extracted and extracted.strip():
                text_per_page[page_num] = extracted.strip()
                continue

            # Fallback to OCR
            print(f"[PDFProcessor] Page {page_num}: Running EasyOCR...")
            ocr_text = self._extract_text_via_ocr(page)
            text_per_page[page_num] = ocr_text.strip()

        return text_per_page

    # ---------------------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------------------
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                meta = reader.metadata

            return {
                "total_pages": len(reader.pages),
                "title": meta.get("/Title", ""),
                "author": meta.get("/Author", ""),
            }
        except Exception:
            return {"total_pages": 0, "title": "", "author": ""}
