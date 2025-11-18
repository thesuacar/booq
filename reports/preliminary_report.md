# BookOutLoud Distributed Architecture – Preliminary Report
**Date:** 2025-11-05 · **Prepared by:** Codex AI Agent

## Objective
Merge the existing AI pipeline (PDF → captions → narration) with the front-end experience by introducing explicit service boundaries. The goal is to let the UI team interact with a stable HTTP API while the AI team owns model code, without tight coupling or direct GPU/model exposure to the interface tier.

## Proposed Topology
```
[Web / Native UI]
       |
       v
[Interface Server (FastAPI)]  <--->  [Object/File Storage]
       |
       v
[Orchestrator Server (FastAPI + asyncio queue)]
       |
       v
[AI Service (FastAPI) → PDF preprocessing, captioning, narration]
```
* A single machine can host all three services for local dev; production can scale them independently.
* All services share the same storage root (`$BOOQ_STORAGE_ROOT`, defaults to `storage/`) so large assets move via the filesystem instead of HTTP payloads.

## Responsibilities
| Server | Key Features | Tech/Ports |
| --- | --- | --- |
| Interface Server (`services/interface_server.py`) | Accepts uploads from the UI, persists PDFs, initiates jobs, proxies status/artifact queries. Never calls AI directly. | FastAPI, HTTP file upload, runs on `:8099` (example). |
| Orchestrator (`services/orchestrator_server.py`) | Persists job metadata, feeds an internal `asyncio.Queue`, calls AI endpoints step-by-step, tracks status/artifacts, exposes `/jobs/**` for the interface. | FastAPI + background worker, uses `httpx` to reach AI service on `:8101`. |
| AI Service (`services/ai_service.py`) | Hosts the actual ML code: PDF text/image extraction, BLIP captioning, gTTS narration + reporting. Provides `/preprocess`, `/caption`, `/narrate` endpoints. | FastAPI, reuses existing `src/*` modules.

## Data & Control Flow
1. UI uploads a PDF to `POST /api/v1/jobs` (interface server). File is stored under `storage/<job-id>/input.pdf`.
2. Interface server registers the job with the orchestrator via `POST /jobs`, passing the job ID, user, language, and path to the stored PDF.
3. Orchestrator enqueues the job and the background worker begins processing: 
   - Calls AI `/preprocess` → writes extracted text + images under `storage/<job-id>/preprocessing/`.
   - Optionally calls `/caption` for extracted images (configurable per job).
   - Builds a narration script (text chunks + “Image N:” annotations) and calls `/narrate`, which outputs MP3 clips and a JSON generation report.
4. Once completed, orchestrator exposes metadata via `GET /jobs/{id}` and `GET /jobs/{id}/artifacts`; the interface server simply forwards these responses to the UI for display/download.

## Integration Guidance
- Front-end only communicates with the interface server. Existing UI upload widgets can point to `http://<host>:8099/api/v1/jobs` and poll `/api/v1/jobs/{id}` for status.
- Environment variables (`BOOQ_ORCHESTRATOR_URL`, `BOOQ_AI_SERVICE_URL`, `BOOQ_STORAGE_ROOT`) keep deployments flexible (Docker compose, single host, or cloud VMs).
- The AI service reuses current modules with minimal change, so modeling teams can continue iterating inside `src/` without touching the interface/orchestrator contracts.

## Next Steps
1. Containerize each service (or add `uvicorn` entry points) and define a `docker-compose.yml` for local orchestration.
2. Add persistence (Redis/Postgres/RabbitMQ) to the orchestrator if cross-instance resiliency is required.
3. Extend the interface server with authentication + signed download URLs so user libraries remain secure.
4. Expand automated tests: endpoint contract tests for each service plus integration tests that spin up the trio via `TestClient`.
