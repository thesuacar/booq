from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import fitz  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/audio"))
sys.path.insert(0, str(ROOT / "src/preprocessing"))
sys.path.insert(0, str(ROOT / "image_captioning"))

import audio_engine  # type: ignore


@pytest.fixture(scope="session")
def data_dir() -> Path:
    path = ROOT / "tests" / "data"
    if not path.exists():
        raise FileNotFoundError("Expected test data directory is missing.")
    return path


def _analyze_pdf(path: Path) -> tuple[bool, bool]:
    has_text = False
    has_images = False
    with fitz.open(path) as doc:
        for page in doc:
            if not has_text and page.get_text("text").strip():
                has_text = True
            if not has_images and page.get_images(full=True):
                has_images = True
            if has_text and has_images:
                break
    return has_text, has_images


@pytest.fixture(scope="session")
def pdf_catalog(data_dir: Path) -> List[dict]:
    catalog: List[dict] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        has_text, has_images = _analyze_pdf(pdf_path)
        catalog.append(
            {
                "path": pdf_path,
                "has_text": has_text,
                "has_images": has_images,
            }
        )

    if not catalog:
        raise FileNotFoundError(
            "tests/data contains no PDFs; add your sample files to exercise the suite."
        )
    return catalog


@pytest.fixture
def stub_gtts(monkeypatch: pytest.MonkeyPatch):
    """Patch gTTS to avoid network calls and capture invocation details."""
    calls: List[dict] = []

    class _DummyTTS:
        def __init__(self, text: str, lang: str):
            calls.append({"text": text, "lang": lang})
            self.text = text
            self.lang = lang

        def save(self, filename: str) -> None:
            Path(filename).write_bytes(b"fake audio data")

    monkeypatch.setattr(audio_engine, "gTTS", _DummyTTS)
    return calls


@pytest.fixture
def stub_captioner(monkeypatch: pytest.MonkeyPatch):
    """Patch ImageCaptioner to skip heavy model loading."""

    class _DummyCaptioner:
        def generate_batch(self, image_paths):
            return [f"caption for {Path(p).name}" for p in image_paths]

    monkeypatch.setattr("run.ImageCaptioner", _DummyCaptioner)
    return _DummyCaptioner
