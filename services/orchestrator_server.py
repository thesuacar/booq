"""Orchestrator FastAPI service coordinating interface requests and AI workers."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import contextlib

STORAGE_ROOT = Path(os.getenv("BOOQ_STORAGE_ROOT", "storage"))
AI_SERVICE_BASE_URL = os.getenv("BOOQ_AI_SERVICE_URL", "http://localhost:8101")


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    user_id: str
    pdf_path: str
    language: str
    original_filename: Optional[str]
    options: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "language": self.language,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "error": self.error,
        }


class JobCreateRequest(BaseModel):
    job_id: str
    user_id: str
    pdf_path: str
    language: str = "en"
    original_filename: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="booq-orchestrator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BOOQ_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, JobRecord] = {}
job_queue: "asyncio.Queue[str]" = asyncio.Queue()
worker_task: Optional[asyncio.Task[Any]] = None


@app.on_event("startup")
async def on_startup() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    global worker_task
    worker_task = asyncio.create_task(job_worker())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if worker_task:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


@app.get("/health")
async def healthcheck() -> Dict[str, Any]:
    return {"status": "ok", "queued_jobs": job_queue.qsize()}


@app.post("/jobs", status_code=202)
async def enqueue_job(payload: JobCreateRequest) -> Dict[str, Any]:
    if payload.job_id in jobs:
        raise HTTPException(status_code=409, detail="Job already exists")
    record = JobRecord(
        job_id=payload.job_id,
        user_id=payload.user_id,
        pdf_path=payload.pdf_path,
        language=payload.language,
        original_filename=payload.original_filename,
        options=payload.options,
    )
    jobs[record.job_id] = record
    await job_queue.put(record.job_id)
    return {"job_id": record.job_id, "status": record.status.value}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict()


@app.get("/jobs/{job_id}/artifacts")
async def get_job_artifacts(job_id: str) -> Dict[str, Any]:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Artifacts only available after completion")
    return record.artifacts


async def job_worker() -> None:
    while True:
        job_id = await job_queue.get()
        record = jobs.get(job_id)
        if not record:
            job_queue.task_done()
            continue
        try:
            await process_job(record)
        except Exception as exc:  # pragma: no cover - operational safeguard
            record.status = JobStatus.FAILED
            record.error = str(exc)
            record.updated_at = datetime.utcnow()
        finally:
            job_queue.task_done()


async def process_job(record: JobRecord) -> None:
    record.status = JobStatus.PROCESSING
    record.updated_at = datetime.utcnow()

    async with httpx.AsyncClient(timeout=120.0) as client:
        preprocess_payload = {
            "job_id": record.job_id,
            "pdf_path": record.pdf_path,
            "output_dir": str(_job_dir(record.job_id) / "preprocessing"),
        }
        preprocess_resp = await _post(client, "/preprocess", preprocess_payload)
        text_path = Path(preprocess_resp["text_path"])
        image_paths = preprocess_resp.get("image_paths", [])

        captions: List[str] = []
        if record.options.get("create_image_captions", True) and image_paths:
            caption_payload = {
                "job_id": record.job_id,
                "image_paths": image_paths,
            }
            caption_resp = await _post(client, "/caption", caption_payload)
            captions = caption_resp.get("captions", [])

        script = _build_script(text_path, captions)
        narration_payload = {
            "job_id": record.job_id,
            "script": script,
            "output_dir": str(_job_dir(record.job_id) / "audio"),
            "language": record.language,
        }
        narration_resp = await _post(client, "/narrate", narration_payload)

    record.status = JobStatus.COMPLETED
    record.updated_at = datetime.utcnow()
    record.artifacts = {
        "text_path": preprocess_resp["text_path"],
        "image_paths": image_paths,
        "captions": captions,
        "audio_files": narration_resp.get("audio_files", []),
        "report_path": narration_resp.get("report_path"),
    }


def _job_dir(job_id: str) -> Path:
    path = STORAGE_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _post(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Dict[str, Any],
    *,
    max_attempts: int = 5,
    retry_delay: float = 2.0,
) -> Dict[str, Any]:
    """POST helper with retries so slow AI startups don't immediately fail jobs."""
    url = f"{AI_SERVICE_BASE_URL}{endpoint}"
    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=payload)
            if response.is_error:
                raise RuntimeError(f"AI service error: {response.status_code} {response.text}")
            return response.json()
        except httpx.RequestError as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                raise RuntimeError(f"AI service unreachable after {max_attempts} attempts: {last_error}") from exc
            await asyncio.sleep(retry_delay)
    raise RuntimeError(f"AI service unreachable: {last_error or 'unknown error'}")


def _build_script(text_path: Path, captions: List[str], max_chars: int = 450) -> List[str]:
    text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    parts: List[str] = []
    for paragraph in (p.strip() for p in text.splitlines() if p.strip()):
        while paragraph:
            parts.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
    for idx, caption in enumerate(captions, start=1):
        parts.append(f"Image {idx}: {caption}")
    return parts or ["No readable text was extracted from the PDF."]
