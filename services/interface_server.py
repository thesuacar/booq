"""
FastAPI Interface Server

Responsibilities:
- Receive requests from UI / external clients
- Persist raw assets (PDFs) under SHARED_STORAGE_ROOT
- Forward audiobook creation requests to the orchestrator server
- Proxy voice listing, health, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    APP_NAME,
    API_PREFIX,
    ORCHESTRATOR_URL,
    SHARED_STORAGE_ROOT,
    PDF_DIR,
    MAX_FILE_SIZE_MB,
    ALLOWED_FILE_TYPES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=f"{APP_NAME} - Interface Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_root = Path(SHARED_STORAGE_ROOT)
storage_root.mkdir(parents=True, exist_ok=True)
pdf_dir = Path(PDF_DIR)
pdf_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CreateAudiobookRequest(BaseModel):
    book_id: str
    pdf_path: str          # absolute or shared path
    language: str          # e.g. "English"
    voice: str | None = None
    speed: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_to_orchestrator(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        url = f"{ORCHESTRATOR_URL}{path}"
        resp = requests.post(url, json=payload, timeout=600)
    except Exception as exc:
        logger.error("Error contacting orchestrator: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to contact orchestrator")

    if resp.status_code != 200:
        logger.error("Orchestrator error %s: %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=resp.status_code, detail=f"Orchestrator error: {resp.text}"
        )

    try:
        return resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from orchestrator")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get(f"{API_PREFIX}/voices")
def list_voices() -> Dict[str, Any]:
    """Proxy to orchestrator /voices."""
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}/voices", timeout=5)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Orchestrator error: {resp.text}",
            )
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error contacting orchestrator for voices: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to contact orchestrator")


@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Receive PDF file from clients, save under shared storage, return its path."""
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail="File too large")

    pdf_path = pdf_dir / file.filename
    with pdf_path.open("wb") as f:
        f.write(contents)

    logger.info("Uploaded PDF saved at %s", pdf_path)
    return {"pdf_path": str(pdf_path)}


@app.post(f"{API_PREFIX}/audiobooks/create")
def create_audiobook(request: CreateAudiobookRequest) -> Dict[str, Any]:
    """
    Public-facing endpoint for audiobook creation.
    Forwards the work order to the orchestrator.
    """
    logger.info(
        "Interface received audiobook create: book_id=%s pdf=%s",
        request.book_id,
        request.pdf_path,
    )

    payload = {
        "book_id": request.book_id,
        "pdf_path": request.pdf_path,
        "language": request.language,
        "voice": request.voice,
        "speed": request.speed,
    }

    result = _forward_to_orchestrator("/audiobooks/create", payload)
    return result
