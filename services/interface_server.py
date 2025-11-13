"""FastAPI server that exposes the public API used by the UI/front-end.

Responsibilities:
- Receive uploads/requests from clients.
- Persist raw assets under the shared storage root.
- Forward work orders to the orchestrator server (never to the AI services directly).
- Proxy status/artifact queries so the UI does not need to know about downstream hosts.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

API_PREFIX = "/api/v1"
STORAGE_ROOT = Path(os.getenv("BOOQ_STORAGE_ROOT", "storage"))
ORCHESTRATOR_BASE_URL = os.getenv("BOOQ_ORCHESTRATOR_URL", "http://localhost:8100")


class OrchestratorClient:
    """Thin async HTTP client to talk to the orchestrator server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(method, url, **kwargs)
        if response.is_error:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

    async def register_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/jobs", json=payload)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}")

    async def get_artifacts(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/jobs/{job_id}/artifacts")


app = FastAPI(title="booq-interface-server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BOOQ_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator_client = OrchestratorClient(ORCHESTRATOR_BASE_URL)


@app.on_event("startup")
def ensure_storage_root() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def healthcheck() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/jobs")
async def submit_book(
    pdf: UploadFile = File(...),
    user_id: str = Form(...),
    language: str = Form("en"),
    create_image_captions: bool = Form(True),
) -> Dict[str, Any]:
    """Entry point invoked by the UI when a user uploads a book."""

    job_id = uuid.uuid4().hex
    job_dir = STORAGE_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "input.pdf"
    contents = await pdf.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    pdf_path.write_bytes(contents)

    payload = {
        "job_id": job_id,
        "user_id": user_id,
        "language": language,
        "pdf_path": str(pdf_path),
        "original_filename": pdf.filename,
        "options": {"create_image_captions": create_image_captions},
    }

    try:
        await orchestrator_client.register_job(payload)
    except HTTPException:
        # Bubble up orchestrator errors directly.
        raise
    except Exception as exc:  # pragma: no cover - network failure paths
        raise HTTPException(status_code=502, detail=str(exc))

    return {"job_id": job_id}


@app.get(f"{API_PREFIX}/jobs/{{job_id}}")
async def get_job(job_id: str) -> Dict[str, Any]:
    try:
        return await orchestrator_client.get_job(job_id)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc))


@app.get(f"{API_PREFIX}/jobs/{{job_id}}/artifacts")
async def get_job_artifacts(job_id: str) -> Dict[str, Any]:
    try:
        return await orchestrator_client.get_artifacts(job_id)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=str(exc))
