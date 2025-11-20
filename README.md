# README

## Project Overview

This repository implements an end‑to‑end audiobook creation system (“booq – Audiobook Creator”) aimed at accessible listening for children and users with visual processing difficulties. The system lets users upload PDFs via a Streamlit web UI, sends jobs to an orchestrator service, converts text and images to audio with a pluggable audio engine, and manages the resulting audiobooks in a local library with playback and personalization options.

Core components:

- Streamlit front‑end (`app/audiobook_app.py`)
- Orchestrator API (`services/orchestrator_server.py`)
- Audio engine / TTS layer (`src/audio_engine.py`)
- Audiobook orchestration and metadata management (`services/audiobook_manager.py`, `src/utils/pdf_utils.py`, `src/utils/preprocessing_utils.py`, `src/utils/captioning_utils.py`)
- AI integration utilities (`services/ai_service.py`, `src/training/encoders.py`, `src/training/model_training.py`, `src/training/training_utils.py`, `src/training/evaluation.py`)
- Configuration and reports (`config.py`, `testing/audio_generation_report.json`)
- Tests (`testing/test_audio_engine.py`, `testing/test_pdf_utils.py`, `testing/test_pipeline_integration.py`, `testing/conftest.py`)

Only the pieces that are directly wired into the running system are described in detail below.

***

## Architecture

### High‑level flow

1. User accesses the Streamlit UI and logs in.
2. User uploads a PDF.
3. The UI stores the PDF on disk and calls the orchestrator HTTP API with `bookid`, `pdfpath`.
4. The orchestrator uses `AudiobookManager` and `AudioEngine` to:
    - Extract text from the PDF
    - Pre‑process text
    - Generate audio via TTS / AI model
    - Save audio and text files
    - Return paths, duration and metadata to the UI.
5. The UI saves this information into a JSON library file and exposes playback, progress tracking and settings.

***

## Components

### Streamlit UI (`audiobook_app.py`)

This is the main user‑facing application built with Streamlit.

Key features:

- **Authentication (simple demo)**: Username/password stored in session state to gate access to the library.
- **Accessible UI**: Large fonts, high‑contrast colors, big buttons and inputs, with sizes and colors configured via `config.py` (font sizes, button heights, color palette, etc.).
- **Library page**:
    - Lists audiobooks from a JSON library file (`LIBRARYFILE`).
    - Search by title, delete books (and associated audio/PDF files), and open a book in the player.
- **Add Book page**:
    - Upload PDF (size/type constrained by `MAXFILESIZEMB`, `ALLOWEDFILETYPES`).
    - On “Create Audiobook”, saves the PDF under `PDFDIR`, calls `ORCHESTRATORURL/audiobooks/create`, and shows progress messages.
- **Player page**:
    - Inline PDF preview using an embedded viewer, when the file exists.
    - HTML5 audio player for the generated audio file (base64 embedded).
    - Playback speed slider using values from `PLAYBACKSPEEDS`.
    - Progress bar showing listening progress and book metadata (duration, pages, voice, generation speed).
- **Settings page**:
    - Default voice and playback speed for new audiobooks, persisted in session state.
    - Values influence future creation/playback but do not retroactively change existing files.

Supporting helper functions include:

- `initsessionstate()` – initializes authentication, library and user settings.
- `loadlibrary() / savelibrary()` – JSON‑based persistence under `DATADIR` and `LIBRARYFILE`.
- `showpdfinline()` – embeds PDFs with base64 HTML in Streamlit.
- `getbackendvoices()` – HTTP GET to `/voices` on the orchestrator.

***

### Configuration (`config.py`)

Central configuration and constants for both UI and backend.

Highlights:

- **App metadata**: `APPNAME`, `APPVERSION`, `APPICON`.
- **Backend endpoint**: `ORCHESTRATORURL` (e.g., `http://localhost:8001`).
- **Directories**: `DATADIR`, `PDFDIR`, `AUDIODIR`, `TEXTDIR`, `CACHEDIR`, `LIBRARYFILE`.
- **Voices and playback speeds**: `VOICES`, `PLAYBACKSPEEDS`, `DEFAULTVOICE`, `DEFAULTSPEED`.
- **Accessibility**: Font sizes for headings/buttons, min button height, etc.
- **File upload limits**: `MAXFILESIZEMB`, `ALLOWEDFILETYPES` (PDF only).
- **Audio generation**: Average words per minute and characters per word for duration estimation.
- **Cache, session and feature flags**: e.g., `ENABLECACHE`, `SESSIONTIMEOUTMINUTES`, `FEATURES` dict toggling capabilities such as search, delete confirmation, speed control and PDF preview.
- **Error messages**: Centralized, user‑friendly error strings used in the UI and backend.

***

### Orchestrator API (`orchestrator_server.py`)

`orchestrator_server.py` exposes a FastAPI service that the UI calls for actual audiobook generation.

Key responsibilities:

- Create core directories (`PDFDIR`, `AUDIODIR`, `TEXTDIR`, plus an audio cache directory).
- Initialize `AudioEngine` (TTS abstraction) and `AudiobookManager` (orchestration logic).
- Endpoints:
    - `GET /health` – simple health check returning a status.
    - `GET /voices` – returns available voice names from `AudioEngine` (used to populate dropdowns in the UI).
    - `POST /audiobooks/create` – main pipeline entry:
        - Validates that the provided `pdfpath` exists.
        - Assembles `AudiobookSettings` from voice and speed.
        - Calls `manager.create_audiobook(...)`.
        - Returns success flag, audio path, text path, duration and total pages, or error details on failure.

The orchestration itself is intentionally thin and delegates PDF processing and audio creation to the services layer to keep endpoint logic small and testable.

***

### Audiobook logic and PDF processing

These modules implement the pipeline used by the orchestrator:

- **`audiobook_manager.py`**:
Encapsulates the audiobook creation workflow around a PDF:
    - Uses the functions in `src/utils/pdf_utils.py` to extract text, images, and metadata per page.
    - Applies any pre‑processing from `preprocessing_utils.py` (cleaning, normalization, splitting).
    - Calls `AudioEngine` to generate audio, including voice selection, speed adjustments, caching.
    - Manages output file paths inside `AUDIODIR` and `TEXTDIR`, and composes a result structure with duration, total pages and any additional metadata.
- **`pdf_utils.py`**:
Provide utilities for page‑wise text extraction (with optional OCR fallback), metadata access, and page rendering via PyMuPDF. These helpers are exercised by the unit tests and power both the FastAPI preprocess endpoint and the audiobook orchestrator.
- **`preprocessing_utils.py`**:
Contains helpers for splitting long texts, cleaning content and preparing it for TTS models (e.g., chunk sizes based on `MODELCONFIG`).
- **`captioning_utils.py`**:
Tools for generating or aligning captions/segments with audio, useful for highlighting listening progress or per‑page mapping (optional layer used by the orchestrator/manager).
    - The default image captioning backend now loads the fine-tuned `epoch_3.pth` checkpoint from `src/training`.
The file `orchestrator-code.py` mirrors these responsibilities in a notebook‑style script (creating `PDFProcessor`, `AudioGenerator` and `AudiobookManager` classes and saving an `.ipynb` file), and documents how to integrate with the Streamlit UI; in this codebase the FastAPI server is the productionized version of that logic.

***

### Audio engine and AI service

- **`audio_engine.py`**:
Abstracts over the underlying TTS implementation and manages a cache directory to avoid re‑generating identical audio.
Typical responsibilities:
    - Track `available_voices` (used by `/voices` endpoint).
    - Given text, voice and speed, either return a cached file or invoke the model/3rd‑party TTS to synthesize new audio.
    - Provide paths or bytes back to `AudiobookManager`.
- **`ai_service.py`**:
Wraps external or local AI models used for TTS or embeddings:
    - Exposes higher‑level `generate_audio(...)` style calls.
    - Uses helper encoders in `encoders.py` and can be trained/configured via utilities in `model_training.py` and `training_utils.py`.
- **`encoders.py`, `model_training.py`, `training_utils.py`, `evaluation.py`**:
Focused on model experimentation and offline training/evaluation rather than runtime path, but share configuration and data conventions with `AudioEngine`/`AudiobookManager`.

***

### Testing

- **`test_audio_engine.py`**:
Tests core behaviors of `AudioEngine`, such as voice listing, caching behavior, and basic generation contract.
- **`test_pdf_utils.py`**:
Verifies PDF extraction helpers against sample PDFs, ensuring page counts and extraction API remain stable.
- **`test_pipeline_integration.py`**:
End‑to‑end or integration tests for the pipeline combining PDF processing, audio engine and audiobook manager logic to ensure a full run works with realistic inputs.
- **`conftest.py`**:
Shared pytest fixtures, e.g., temporary directories, sample PDFs, stubbed audio engines or orchestrator instances.

`pytest_output.txt` captures a sample test run, useful to see current pass/fail status but not needed for normal usage.

***

## Installation

### Prerequisites

- Python 3.9+ and < 3.14.
- `pip` and a virtual environment tool (e.g., `venv` or `conda`).
- For full TTS functionality, additional libraries such as PyTorch, transformers or gTTS, depending on how `AudioEngine` and `ai_service` are configured via `MODELCONFIG`.


### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>.git
cd <repo-folder>
```

2. **Create and activate a virtual environment**
```bash
python3.13 -m venv SEenv

source SEenv/bin/activate  # Windows: SEenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r booq_requirements.txt
```

4. **Create data directories**

The app will attempt to create them automatically, but you can ensure they exist:

```bash
mkdir -p data data/pdfs data/audio data/text data/audio/cache 
```

```bash
mkdir data, data/pdfs, data/audio, data/audio_cache # for Windows
```

These locations can be changed in `config.py` if needed.

***

## Running the System

### 1. Start the orchestrator server

From the project root:

```bash
uvicorn booq.services.orchestrator_server:app --reload --host 0.0.0.0 --port 8001
```

This must match `ORCHESTRATORURL` in `config.py` (by default pointing at `http://localhost:8001`).

Check the health endpoint:

```bash
curl http://localhost:8001/health
```

You should see a small JSON status response.

### 2. Start the Streamlit UI

In a separate terminal with the same virtual environment:

```bash
streamlit run audiobook_app.py
```

This launches the accessible web UI described above.

***

## Using the Application

1. **Open the UI**
Navigate to the URL printed by Streamlit (typically `http://localhost:8501`).
2. **Log in**
Enter any username and password (demo auth) and click “Login” to access your library.
3. **Add a new audiobook**
    - Go to **Add Book**.
    - Upload a PDF file (must be a `.pdf` and under `MAXFILESIZEMB` MB).
    - Adjust default playback speed and voice in **Settings** first if desired.
    - Click **Create Audiobook**:
        - The UI saves the PDF to `PDFDIR` and calls the orchestrator.
        - A progress bar and status text show extraction, audio generation and saving steps.
    - When finished, you will see success messages and can either:
        - Click **Play Now** to go directly to the player, or
        - Return to the **Library**.
4. **Browse and manage the library**
    - The **Library** view lists all created audiobooks from `datalibrary.json`, showing title, date added, status and duration.
    - Use the search box to filter by title.
    - Click **Play** to open the player page for a book.
    - Click **Delete** to remove a book; the app will ask for confirmation and delete associated audio/PDF files as well as the entry in the JSON library.
5. **Listen to an audiobook**
    - The player page shows:
        - Book title and metadata.
        - PDF preview if the source file is still present.
        - HTML5 audio player with adjustable playback speed, backed by `PLAYBACKSPEEDS`.
        - Listening progress bar and percentage.
    - The voice selector here affects future generations for that book or new books, depending on how you save settings.
6. **Change settings**
    - In **Settings**, choose your default voice and default playback speed.
    - Click **Save Settings**; these are stored in session state and used as defaults for subsequent audiobook creation/playback.

***

## Notes and Extensibility

- `MODELCONFIG` in `config.py` allows swapping between different model types (custom, gTTS, transformers, etc.), and `AudioEngine`/`ai_service` can be extended to use any TTS backend.
- `audio_generation_report.json` can store information about audio runs (durations, errors, metadata) for analysis or debugging but is not required for normal operation.
- The notebook‑style `orchestrator-code.py` documents the same architecture and can be used for experimentation or as an explanatory reference.


***


