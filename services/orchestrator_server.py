"""Orchestrator FastAPI service coordinating interface requests and AI workers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .audiobook_manager import AudiobookManager, AudiobookSettings

app = FastAPI(
    title="booq Orchestrator Service",
    description="Orchestrator between UI and AI audiobook pipeline",
)

manager = AudiobookManager()


class CreateAudiobookRequest(BaseModel):
    book_id: str
    pdf_path: str
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0


class CreateAudiobookResponse(BaseModel):
    success: bool
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    metadata: Optional[dict] = None
    total_pages: Optional[int] = None
    duration: Optional[str] = None
    error: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/audiobooks/create", response_model=CreateAudiobookResponse)
def create_audiobook(payload: CreateAudiobookRequest):
    """
    Create an audiobook from a PDF.

    This is called by the Streamlit UI.
    """
    pdf_path = Path(payload.pdf_path)

    settings = AudiobookSettings(
        language=payload.language,
        voice=payload.voice,
        speed=payload.speed,
    )

    # For now we call synchronously; future: background tasks, job queue, etc.
    result = manager.create_audiobook(
        pdf_path=pdf_path,
        book_id=payload.book_id,
        settings=settings,
        progress_callback=None,  # For HTTP we omit live progress
    )

    return CreateAudiobookResponse(**result)
