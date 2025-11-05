"""End-to-end test pipeline for the BOOQ backend system."""

from pathlib import Path
from pdf_utils import extract_txt_from_pdf, extract_images_from_pdf
from image_captioning import ImageCaptioner
from audio_engine import synthesize_captions, export_generation_report


def generate_audio_from_pdf(
    pdf_path: Path,
    output_dir: Path = Path("outputs"),
    lang: str = "en",
    include_text: bool = True,
    include_images: bool = True,
) -> dict:
    """Full pipeline: PDF → text/images → captions → audio → report."""
    # Prepare output directories
    text_dir = output_dir / "text"
    image_dir = output_dir / "images"
    audio_dir = output_dir / "audio"
    text_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_captions = []

    # --- 1️⃣ Extract text ---
    if include_text:
        print("Extracting text from PDF...")
        text = extract_txt_from_pdf(pdf_path)
        text_output = text_dir / "extracted_text.txt"
        text_output.write_text(text, encoding="utf-8")

        # Split into short chunks for TTS
        text_captions = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
        all_captions.extend(text_captions)
        print(f"✓ Extracted {len(text_captions)} text captions.")

    # --- 2️⃣ Extract images ---
    image_paths = []
    if include_images:
        print("Extracting images from PDF...")
        extract_images_from_pdf(pdf_path, image_dir)
        image_paths = list(image_dir.glob("*"))
        print(f"✓ Extracted {len(image_paths)} images.")

    # --- 3️⃣ Generate captions for images ---
    if image_paths:
        print("Generating image captions...")
        captioner = ImageCaptioner()
        image_captions = captioner.generate_batch(image_paths)
        all_captions.extend(image_captions)
        print(f"✓ Generated {len(image_captions)} image captions.")

    # --- 4️⃣ Synthesize audio ---
    print("Generating audio narration...")
    clips = synthesize_captions(all_captions, output_dir=audio_dir, lang=lang)
    print(f"✓ Generated {len(clips)} audio clips.")

    # --- 5️⃣ Export report ---
    print("Saving generation report...")
    report = export_generation_report(
        clips,
        output_path=output_dir / "audio_generation_report.json",
        extra_metadata={
            "source_pdf": str(pdf_path),
            "num_text_captions": len(all_captions),
            "language": lang,
        },
    )

    print("✓ Pipeline complete.")
    return report


if __name__ == "__main__":
    # Example test run
    sample_pdf = Path("samples/book_sample.pdf")  # 🔹 Place a small test PDF here
    output = generate_audio_from_pdf(sample_pdf)
    print("\nPipeline finished. Report summary:")
    print(output)
