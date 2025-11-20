"""AI utilities for generating descriptive captions from extracted images."""

from __future__ import annotations

from pathlib import Path
from typing import List

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class ImageCaptioner:
    """Handles AI-based caption generation for image files."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base") -> None:
        # Load processor and model once (expensive step)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate_caption(self, image_path: Path) -> str:
        """Generate a caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Cannot open image {image_path}: {e}")
            return ""

        inputs = self.processor(images=image, return_tensors="pt")
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def generate_batch(self, image_paths: List[Path]) -> List[str]:
        """Generate captions for a list of image paths."""
        captions = []
        for path in image_paths:
            caption = self.generate_caption(path)
            if caption:
                captions.append(caption)
        return captions
