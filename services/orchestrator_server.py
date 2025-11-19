"""
Orchestrator Server for Audiobook Generation
Bridges Streamlit UI <-> AudiobookManager
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from config import *  # expects DEFAULT_VOICE, DEFAULT_SPEED, PDF_DIR, AUDIO_DIR, TEXT_DIR

from src.audio_engine import AudioEngine
from services.audiobook_manager import AudiobookManager, AudiobookSettings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="booq-orchestrator-server", version="0.1.0")

# -----------------------------------------------------------------------------
# Backend components
# -----------------------------------------------------------------------------

# Make sure the traditional directories still exist (even if the manager uses its own root)
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
Path(TEXT_DIR).mkdir(parents=True, exist_ok=True)

# Audio engine used both by the manager and the /voices endpoint
audio_cache_dir = Path(AUDIO_DIR) / "cache"
audio_cache_dir.mkdir(parents=True, exist_ok=True)
audio_engine = AudioEngine(cache_dir=audio_cache_dir)

# High-level audiobook orchestrator
manager = AudiobookManager(audio_engine=audio_engine, pdf_processor=None)

logger.info("--- Orchestrator Started ---")
logger.info("Voices available: %s", list(audio_engine.available_voices.keys()))


# -----------------------------------------------------------------------------
# Request schema
# -----------------------------------------------------------------------------

class CreateAudiobookRequest(BaseModel):
    book_id: str
    pdf_path: str
    language: str   # e.g. "English", "Spanish", "en", "es"
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _create_audiobook_from_payload(payload: CreateAudiobookRequest) -> dict:
    """
    Thin wrapper around AudiobookManager so the endpoint logic stays tiny.
    """
    settings = AudiobookSettings(
        language=payload.language,
        voice=payload.voice,
        speed=payload.speed,
    )

    result = manager.create_audiobook(
        pdf_path=payload.pdf_path,
        book_id=payload.book_id,
        settings=settings,
    )

    if not result.get("success"):
        # Already includes an "error" field
        return {
            "success": False,
            "error": result.get("error", "Unknown error"),
        }

    return {
        "success": True,
        "audio_path": result["audio_path"],
        "text_path": result["text_path"],
        "duration": result["duration"],
        "total_pages": result.get("total_pages", 0),
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/voices")
def list_voices() -> dict:
    """
    Returns the voice names the UI can show in a dropdown.
    """
    try:
        return {"voices": list(audio_engine.available_voices.keys())}
    except Exception as exc:
        logger.error("Error listing voices: %s", exc)
        return {"voices": [], "error": str(exc)}


@app.post("/audiobooks/create")
def create_audiobook(payload: CreateAudiobookRequest) -> dict:
    """
    Full pipeline entry point:

    PDF path (already stored on disk) →
        AudiobookManager (PDF → text → audio) →
        return paths and metadata to the caller.
    """
    try:
        logger.info(
            "Starting audiobook job: book_id=%s, pdf=%s, lang=%s, voice=%s, speed=%.2f",
            payload.book_id,
            payload.pdf_path,
            payload.language,
            payload.voice,
            payload.speed,
        )

        if not Path(payload.pdf_path).exists():
            return {"success": False, "error": f"PDF not found: {payload.pdf_path}"}

        result = _create_audiobook_from_payload(payload)

        if result.get("success"):
            logger.info("Audiobook created successfully for book_id=%s", payload.book_id)
        else:
            logger.error(
                "Audiobook creation failed for book_id=%s: %s",
                payload.book_id,
                result.get("error"),
            )

        return result

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Exception during audiobook creation: %s\n%s", exc, tb)
        return {"success": False, "error": str(exc), "traceback": tb}
