import run
import pytest
from pathlib import Path


def test_generate_audio_from_pdf_text_only_pipeline(
    pdf_catalog, tmp_path, stub_gtts
) -> None:
    for entry in pdf_catalog:
        stub_gtts.clear()
        output_dir = tmp_path / f"{entry['path'].stem}_text_only"
        report = run.generate_audio_from_pdf(
            entry["path"],
            output_dir=output_dir,
            include_images=False,
        )

        assert len(report["clips"]) == len(stub_gtts) == report["num_text_captions"]
        if entry["has_text"]:
            assert len(stub_gtts) > 0, f"Expected text captions for {entry['path']}"

        for clip in report["clips"]:
            assert Path(clip["file"]).exists()


def test_generate_audio_from_pdf_images_only_pipeline(
    pdf_catalog, tmp_path, stub_gtts, stub_captioner
) -> None:
    image_docs = [entry for entry in pdf_catalog if entry["has_images"]]
    if not image_docs:
        pytest.skip("No PDFs with embedded images found in tests/data.")

    for entry in image_docs:
        stub_gtts.clear()
        output_dir = tmp_path / f"{entry['path'].stem}_images_only"
        report = run.generate_audio_from_pdf(
            entry["path"],
            output_dir=output_dir,
            include_text=False,
        )

        assert len(report["clips"]) == len(stub_gtts) == report["num_text_captions"]
        assert all("caption for" in clip["caption"] for clip in report["clips"])


def test_generate_audio_from_pdf_combined_pipeline(
    pdf_catalog, tmp_path, stub_gtts, stub_captioner
) -> None:
    for entry in pdf_catalog:
        stub_gtts.clear()
        output_dir = tmp_path / f"{entry['path'].stem}_combined"
        report = run.generate_audio_from_pdf(
            entry["path"],
            output_dir=output_dir,
            lang="es",
        )

        assert len(report["clips"]) == len(stub_gtts) == report["num_text_captions"]
        if entry["has_text"]:
            assert any(
                not call["text"].startswith("caption for") for call in stub_gtts
            ), f"Expected textual captions for {entry['path']}"
        if entry["has_images"]:
            assert any(
                call["text"].startswith("caption for") for call in stub_gtts
            ), f"Expected image captions for {entry['path']}"
        assert all(call["lang"] == "es" for call in stub_gtts)
