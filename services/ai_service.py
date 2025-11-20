"""AI microservice hosting preprocessing, captioning, and narration endpoints."""

from __future__ import annotations

import os
from array import array
from functools import lru_cache
import math
from pathlib import Path
from typing import Dict, List, Optional
import wave

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.audio_engine import AudioClip, export_generation_report, synthesize_captions
from src.utils.captioning_utils import ImageCaptioner
from src.utils.pdf_utils import extract_images_from_pdf, extract_txt_from_pdf

STORAGE_ROOT = Path(os.getenv("BOOQ_STORAGE_ROOT", "storage"))
USE_FAKE_TTS = os.getenv("BOOQ_USE_FAKE_TTS", "true").lower() in {"1", "true", "yes"}


class PreprocessRequest(BaseModel):
    job_id: str
    pdf_path: str
    output_dir: str


class PreprocessResponse(BaseModel):
    text_path: str
    image_paths: List[str]


class CaptionRequest(BaseModel):
    job_id: str
    image_paths: List[str]
    limit: Optional[int] = None


class CaptionResponse(BaseModel):
    captions: List[str]


class NarrateRequest(BaseModel):
    job_id: str
    script: List[str]
    output_dir: str
    language: str = "en"


class NarrateResponse(BaseModel):
    audio_files: List[str]
    report_path: str


app = FastAPI(title="booq-ai-service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BOOQ_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_pdf(payload: PreprocessRequest) -> PreprocessResponse:
    pdf_path = Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    out_dir = Path(payload.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_content = extract_txt_from_pdf(str(pdf_path))
    text_path = out_dir / "book.txt"
    text_path.write_text(text_content, encoding="utf-8")

    images_dir = out_dir / "images"
    image_paths = extract_images_from_pdf(str(pdf_path), str(images_dir))

    return PreprocessResponse(text_path=str(text_path), image_paths=image_paths)


@app.post("/caption", response_model=CaptionResponse)
def caption_images(payload: CaptionRequest) -> CaptionResponse:
    if not payload.image_paths:
        return CaptionResponse(captions=[])

    captioner = _get_captioner()
    paths = [Path(p) for p in payload.image_paths if Path(p).exists()]
    if not paths:
        return CaptionResponse(captions=[])

    captions = captioner.generate_batch(paths[: payload.limit] if payload.limit else paths)
    return CaptionResponse(captions=captions)


@app.post("/narrate", response_model=NarrateResponse)
def narrate_script(payload: NarrateRequest) -> NarrateResponse:
    if not payload.script:
        raise HTTPException(status_code=400, detail="Script is empty")

    audio_dir = Path(payload.output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if USE_FAKE_TTS:
        clips = _synthesize_offline_audio(
            payload.script,
            output_dir=audio_dir,
            job_id=payload.job_id,
        )
    else:
        clips = synthesize_captions(
            payload.script,
            output_dir=audio_dir,
            prefix=f"{payload.job_id}_line",
            lang=payload.language,
        )

    report_path = audio_dir / "generation_report.json"
    export_generation_report(
        clips,
        output_path=report_path,
        extra_metadata={"job_id": payload.job_id, "language": payload.language},
    )

    return NarrateResponse(
        audio_files=[str(clip.path) for clip in clips],
        report_path=str(report_path),
    )


@lru_cache(maxsize=1)
def _get_captioner() -> ImageCaptioner:
    return ImageCaptioner()


def _synthesize_offline_audio(script: List[str], *, output_dir: Path, job_id: str) -> List[AudioClip]:
    """Fast, offline-friendly audio generator that emits short WAV clips."""
    sample_rate = 16000
    duration_seconds = 1.2
    amplitude = 12000
    clips: List[AudioClip] = []

    for index, text in enumerate(script, start=1):
        filename = f"{job_id}_line_{index}.wav"
        file_path = output_dir / filename
        total_samples = int(sample_rate * duration_seconds)
        frequency = 440 + (index % 5) * 70
        samples = array(
            "h",
            (
                int(amplitude * math.sin(2 * math.pi * frequency * (sample_index / sample_rate)))
                for sample_index in range(total_samples)
            ),
        )
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        clips.append(AudioClip(caption=text, path=file_path, generation_time=duration_seconds))

    return clips
