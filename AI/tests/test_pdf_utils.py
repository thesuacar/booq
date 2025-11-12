from pathlib import Path

import pytest

from src.preprocessing.pdf_utils import extract_images_from_pdf, extract_txt_from_pdf


def test_extract_txt_from_pdf_returns_text_for_documents_with_text(pdf_catalog) -> None:
    text_docs = [entry for entry in pdf_catalog if entry["has_text"]]
    if not text_docs:
        pytest.skip("No PDFs with textual content found in tests/data.")

    for entry in text_docs:
        text = extract_txt_from_pdf(entry["path"])
        assert text.strip(), f"Expected non-empty text for {entry['path']}"


def test_extract_txt_from_pdf_handles_documents_without_text(pdf_catalog) -> None:
    empty_docs = [entry for entry in pdf_catalog if not entry["has_text"]]
    if not empty_docs:
        pytest.skip("No text-free PDFs found in tests/data.")

    for entry in empty_docs:
        text = extract_txt_from_pdf(entry["path"])
        assert text.strip() == "", f"Expected empty text for {entry['path']}"


def test_extract_images_from_pdf_with_images(pdf_catalog, tmp_path: Path) -> None:
    image_docs = [entry for entry in pdf_catalog if entry["has_images"]]
    if not image_docs:
        pytest.skip("No PDFs containing images found in tests/data.")

    for entry in image_docs:
        image_dir = tmp_path / f"{entry['path'].stem}_images"
        images = extract_images_from_pdf(entry["path"], image_dir)
        assert images, f"Expected extracted images for {entry['path']}"
        for image_path in images:
            assert Path(image_path).exists()


def test_extract_images_from_pdf_without_images(pdf_catalog, tmp_path: Path) -> None:
    non_image_docs = [entry for entry in pdf_catalog if not entry["has_images"]]
    if not non_image_docs:
        pytest.skip("All PDFs contain images; add a sample without embedded images.")

    for entry in non_image_docs:
        image_dir = tmp_path / f"{entry['path'].stem}_images"
        images = extract_images_from_pdf(entry["path"], image_dir)
        assert images == []
        assert image_dir.exists()
