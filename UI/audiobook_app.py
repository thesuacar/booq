
"""
Audiobook Creator Application - Main UI
Built with Streamlit for accessibility (VPD-focused)
This application helps create AI-generated audiobooks from PDF files
"""

import streamlit as st
import os
import json
import shutil
from pathlib import Path
import time
from datetime import datetime
from typing import List, Tuple, Optional

import httpx
import fitz

import config

# Page Configuration
st.set_page_config(
    page_title="booq - Audiobook Creator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for accessibility - Large fonts and high contrast
st.markdown("""
    <style>
    /* Global font size increase for vision-impaired users */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    /* Headers - Extra large */
    h1 {
        font-size: 48px !important;
        font-weight: bold !important;
        margin-bottom: 30px !important;
    }

    h2 {
        font-size: 36px !important;
        font-weight: bold !important;
        margin-bottom: 20px !important;
    }

    h3 {
        font-size: 28px !important;
        font-weight: bold !important;
    }

    /* Button styling - Large and clear */
    .stButton > button {
        font-size: 24px !important;
        padding: 20px 40px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        min-height: 60px !important;
    }

    /* Input fields - Large text */
    .stTextInput > div > div > input {
        font-size: 22px !important;
        padding: 15px !important;
        min-height: 50px !important;
    }

    /* Select boxes */
    .stSelectbox > div > div > select {
        font-size: 22px !important;
        padding: 15px !important;
    }

    /* File uploader */
    .stFileUploader > div {
        font-size: 20px !important;
    }

    /* High contrast for better readability */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }

    /* Card-like containers for books */
    .book-card {
        border: 3px solid #333;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        background-color: #F8F8F8;
        font-size: 20px;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        height: 30px !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        font-size: 20px !important;
    }

    /* Audio player */
    audio {
        width: 100% !important;
        height: 60px !important;
    }

    /* Alert/Info boxes */
    .stAlert {
        font-size: 20px !important;
        padding: 20px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 24px !important;
        padding: 15px 30px !important;
    }
    </style>
""", unsafe_allow_html=True)

API_BASE_URL = f"{config.INTERFACE_BASE_URL.rstrip('/')}{config.INTERFACE_API_PREFIX}"
SESSION_USER_FALLBACK = "guest"


def interface_api_url(path: str) -> str:
    """Build a fully-qualified URL for the interface server."""
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{API_BASE_URL}{suffix}"


def ensure_data_dirs() -> None:
    """Make sure all local data directories exist before writing files."""
    for folder in [
        config.DATA_DIR,
        config.PDF_DIR,
        config.AUDIO_DIR,
        config.CACHE_DIR,
        config.TRANSCRIPTS_DIR,
        config.PREVIEW_DIR,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def submit_job_to_pipeline(pdf_bytes: bytes, filename: str, language_code: str, user_id: str) -> str:
    """Upload a PDF to the interface server and receive a job id."""
    url = interface_api_url("/jobs")
    files = {"pdf": (filename, pdf_bytes, "application/pdf")}
    data = {
        "user_id": user_id or SESSION_USER_FALLBACK,
        "language": language_code,
        "create_image_captions": "true",
    }
    try:
        response = httpx.post(url, data=data, files=files, timeout=120.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Interface error ({exc.response.status_code}): {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Unable to reach interface server: {exc}") from exc

    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError("Interface server did not return a job id.")
    return job_id


def fetch_job_status(job_id: str) -> dict:
    """Fetch current status for a job id."""
    url = interface_api_url(f"/jobs/{job_id}")
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Interface status error ({exc.response.status_code}): {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Unable to reach interface server: {exc}") from exc
    return response.json()


def wait_for_job_completion(job_id: str, status_text, progress_bar) -> dict:
    """Poll the orchestrator until a job finishes or fails."""
    start = time.time()
    while True:
        job_data = fetch_job_status(job_id)
        status = job_data.get("status", "pending").lower()

        if status == "completed":
            progress_bar.progress(98)
            status_text.text("📦 Gathering generated audio...")
            return job_data
        if status == "failed":
            error_message = job_data.get("error", "Unknown error during processing.")
            raise RuntimeError(f"Job failed: {error_message}")

        elapsed = time.time() - start
        if elapsed > config.JOB_POLL_TIMEOUT:
            raise TimeoutError("Timed out waiting for the audiobook generation to finish.")

        progress_value = min(95, int((elapsed / config.JOB_POLL_TIMEOUT) * 90) + 5)
        progress_bar.progress(progress_value)
        status_text.text(f"⏳ Job status: {status.title()}. Checking again soon...")
        time.sleep(config.JOB_POLL_INTERVAL)


def fetch_job_artifacts(job_id: str) -> dict:
    """Retrieve finished artifact metadata from the interface server."""
    url = interface_api_url(f"/jobs/{job_id}/artifacts")
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Artifact fetch error ({exc.response.status_code}): {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Unable to reach interface server: {exc}") from exc
    return response.json()


def persist_artifacts(job_id: str, artifacts: dict) -> Tuple[List[str], str, List[str]]:
    """Copy generated files into the UI's data directory."""
    audio_paths: List[str] = []
    audio_dest_dir = Path(config.AUDIO_DIR) / job_id
    audio_dest_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(artifacts.get("audio_files", []), start=1):
        src_path = Path(src)
        if not src_path.exists():
            continue
        dest = audio_dest_dir / f"{job_id}_{idx}{src_path.suffix}"
        shutil.copy2(src_path, dest)
        audio_paths.append(str(dest))

    transcript_path = None
    text_src = artifacts.get("text_path")
    if text_src:
        text_path = Path(text_src)
        if text_path.exists():
            transcript_dest_dir = Path(config.TRANSCRIPTS_DIR)
            transcript_dest_dir.mkdir(parents=True, exist_ok=True)
            transcript_dest = transcript_dest_dir / f"{job_id}.txt"
            shutil.copy2(text_path, transcript_dest)
            transcript_path = str(transcript_dest)

    captions = artifacts.get("captions", [])
    return audio_paths, transcript_path, captions


def generate_preview_image(pdf_path: Path, book_id: str) -> Optional[str]:
    """Render the first page of a PDF to an image for preview."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        zoom = fitz.Matrix(2, 2)  # higher resolution
        pix = page.get_pixmap(matrix=zoom)
        preview_dir = Path(config.PREVIEW_DIR)
        preview_dir.mkdir(parents=True, exist_ok=True)
        file_path = preview_dir / f"{book_id}.png"
        pix.save(file_path)
        return str(file_path)
    except Exception as exc:
        print(f"[WARN] Unable to generate preview for {pdf_path}: {exc}")
        return None


def safe_remove(path: Path) -> None:
    """Remove a file if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def sanitize_title(filename: str) -> str:
    """Create a display-friendly title from the uploaded filename."""
    return Path(filename).stem.strip() or "Untitled Book"


def map_language_to_code(language: str) -> str:
    """Map UI language names to the backend language codes."""
    return config.LANGUAGE_CODES.get(language, config.LANGUAGE_CODES.get(config.DEFAULT_LANGUAGE, "en"))


def current_user_id() -> str:
    """Return the authenticated username or a safe fallback."""
    return st.session_state.username or SESSION_USER_FALLBACK


# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
    ensure_data_dirs()
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'library' not in st.session_state:
        st.session_state.library = load_library()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'default_voice': 'Ana',
            'default_speed': 1.0,
            'theme': 'light'
        }

# Data persistence functions
def load_library():
    """Load library from JSON file"""
    library_path = Path(config.LIBRARY_FILE)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        with open(library_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_library():
    """Save library to JSON file"""
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    with open(config.LIBRARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.library, f, indent=2, ensure_ascii=False)

# Authentication functions
def login():
    """Simple login page"""
    st.markdown("<h1 style='text-align: center;'>🎧 booq - Audiobook Creator</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Login to Access Your Library</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        username = st.text_input("👤 Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", key="login_password", placeholder="Enter your password")

        st.markdown("<br>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Login", use_container_width=True):
                # Simple authentication - in production, use proper auth
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.current_page = "library"
                    st.rerun()
                else:
                    st.error("⚠️ Please enter both username and password")

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.current_page = "login"
    st.rerun()

# Library page
def library_page():
    """Main library page showing all books"""
    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<h1>📚 My Audiobook Library</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 22px;'>Welcome back, <strong>{st.session_state.username}</strong>!</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    # Search bar
    search_query = st.text_input("🔍 Search books by title", placeholder="Type to search...")

    # Add new book button
    col_add1, col_add2, col_add3 = st.columns([2, 1, 2])
    with col_add2:
        if st.button("➕ Add New Book", use_container_width=True, type="primary"):
            st.session_state.current_page = "add_book"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Display books
    library = st.session_state.library

    if not library:
        st.info("📖 Your library is empty. Click 'Add New Book' to get started!")
    else:
        # Filter books based on search
        filtered_books = {
            book_id: book_data 
            for book_id, book_data in library.items() 
            if search_query.lower() in book_data['title'].lower()
        }

        if not filtered_books:
            st.warning(f"No books found matching '{search_query}'")
        else:
            st.markdown(f"<h3>Found {len(filtered_books)} book(s)</h3>", unsafe_allow_html=True)

            # Display books in a grid
            for book_id, book_data in filtered_books.items():
                with st.container():
                    col_book1, col_book2 = st.columns([4, 1])

                    with col_book1:
                        st.markdown(f"""
                        <div class="book-card">
                            <h2 style="margin: 0;">{book_data['title']}</h2>
                            <p style="font-size: 20px; margin: 10px 0;">
                                🌍 Language: <strong>{book_data['language']}</strong><br>
                                📅 Added: {book_data['date_added']}<br>
                                🎵 Status: <strong>{book_data['status']}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_book2:
                        st.markdown("<br>" * 2, unsafe_allow_html=True)
                        if st.button("▶️ Play", key=f"play_{book_id}", use_container_width=True):
                            st.session_state.selected_book = book_id
                            st.session_state.current_page = "player"
                            st.rerun()

                        if st.button("🗑️ Delete", key=f"delete_{book_id}", use_container_width=True):
                            st.session_state.book_to_delete = book_id
                            st.session_state.show_delete_confirm = True

    # Delete confirmation dialog
    if 'show_delete_confirm' in st.session_state and st.session_state.show_delete_confirm:
        book_id = st.session_state.book_to_delete
        book_title = library[book_id]['title']

        st.markdown("---")
        st.warning(f"⚠️ Are you sure you want to delete '{book_title}'?")

        col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
        with col_del2:
            if st.button("✅ Yes, Delete", use_container_width=True):
                book_data = library[book_id]

                # Delete audio clips and directory
                audio_clips = book_data.get('audio_clips') or []
                for clip in audio_clips:
                    safe_remove(Path(clip))
                audio_dir = Path(config.AUDIO_DIR) / book_data.get('job_id', book_id)
                if audio_dir.exists() and audio_dir.is_dir():
                    shutil.rmtree(audio_dir, ignore_errors=True)

                # Delete transcript and PDF copies
                transcript_path = book_data.get('transcript_path')
                if transcript_path:
                    safe_remove(Path(transcript_path))
                pdf_path = book_data.get('pdf_path')
                if pdf_path:
                    safe_remove(Path(pdf_path))
                preview_path = book_data.get('preview_path')
                if preview_path:
                    safe_remove(Path(preview_path))

                # Remove from library
                del st.session_state.library[book_id]
                save_library()
                st.session_state.show_delete_confirm = False
                st.rerun()

        with col_del3:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_delete_confirm = False
                st.rerun()

# Add book page
def add_book_page():
    """Page for adding a new book"""
    st.markdown("<h1>➕ Add New Audiobook</h1>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Library"):
        st.session_state.current_page = "library"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload PDF
    st.markdown("<h2>1️⃣ Upload Your Book (PDF)</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'],
        help="Upload a PDF book to convert to audiobook"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Language selection
    st.markdown("<h2>2️⃣ Select Audiobook Language</h2>", unsafe_allow_html=True)
    default_lang_index = config.LANGUAGES.index(config.DEFAULT_LANGUAGE) if config.DEFAULT_LANGUAGE in config.LANGUAGES else 0
    language = st.selectbox(
        "Choose language for audiobook",
        options=config.LANGUAGES,
        index=default_lang_index
    )

    st.markdown("<br>" * 2, unsafe_allow_html=True)

    # Create audiobook button
    col_create1, col_create2, col_create3 = st.columns([1, 1, 1])
    with col_create2:
        create_button = st.button("🎙️ Create Audiobook", use_container_width=True, type="primary", disabled=not uploaded_file)

    if create_button and uploaded_file:
        st.markdown("<br>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        pdf_bytes = uploaded_file.getvalue()
        timestamp = int(time.time())
        pdf_filename = f"{timestamp}_{uploaded_file.name}"
        pdf_path = Path(config.PDF_DIR) / pdf_filename
        pdf_path.write_bytes(pdf_bytes)
        book_id = f"book_{timestamp}"
        preview_path = generate_preview_image(pdf_path, book_id)

        try:
            st.info(config.INFO_MESSAGES['creating_audiobook'])
            status_text.text("📤 Uploading book to processing queue...")
            language_code = map_language_to_code(language)
            job_id = submit_job_to_pipeline(pdf_bytes, uploaded_file.name, language_code, current_user_id())
            progress_bar.progress(10)

            status_text.text("🧠 Job queued. Waiting for completion...")
            wait_for_job_completion(job_id, status_text, progress_bar)

            artifacts = fetch_job_artifacts(job_id)
            audio_paths, transcript_path, captions = persist_artifacts(job_id, artifacts)

            book_title = sanitize_title(uploaded_file.name)
            st.session_state.library[book_id] = {
                'title': book_title,
                'language': language,
                'language_code': language_code,
                'date_added': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'pdf_path': str(pdf_path),
                'audio_path': audio_paths[0] if audio_paths else None,
                'audio_clips': audio_paths,
                'transcript_path': transcript_path,
                'captions': captions,
                'report_path': artifacts.get('report_path'),
                'image_paths': artifacts.get('image_paths', []),
                'preview_path': preview_path,
                'status': 'Ready' if audio_paths else 'Needs Attention',
                'progress': 0,
                'duration': 'Unknown',
                'current_page': 1,
                'total_pages': 0,
                'job_id': job_id,
            }
            save_library()

            progress_bar.progress(100)
            status_text.text("✅ Audiobook created successfully!")
            if not audio_paths:
                st.warning("The pipeline completed but no audio files were returned. Check the backend logs.")

            st.success(f"✅ Audiobook '{book_title}' created successfully!")
            st.info("📚 Your book is now available in the library. Click below to start listening!")

            col_play1, col_play2, col_play3 = st.columns([1, 1, 1])
            with col_play2:
                if st.button("▶️ Play Now", use_container_width=True):
                    st.session_state.selected_book = book_id
                    st.session_state.current_page = "player"
                    st.rerun()

            with col_play3:
                if st.button("📚 Go to Library", use_container_width=True):
                    st.session_state.current_page = "library"
                    st.rerun()
        except TimeoutError as exc:
            status_text.text("")
            st.error(f"⏱️ {exc}")
        except RuntimeError as exc:
            status_text.text("")
            st.error(f"❌ {exc}")

# Player page
def player_page():
    """Audio player page for playing books"""
    if not st.session_state.selected_book:
        st.error("No book selected")
        if st.button("⬅️ Back to Library"):
            st.session_state.current_page = "library"
            st.rerun()
        return

    book_id = st.session_state.selected_book
    book_data = st.session_state.library.get(book_id)

    if not book_data:
        st.error("Book not found")
        if st.button("⬅️ Back to Library"):
            st.session_state.current_page = "library"
            st.rerun()
        return

    # Header
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown(f"<h1>🎧 {book_data['title']}</h1>", unsafe_allow_html=True)
    with col_header2:
        if st.button("⬅️ Library", use_container_width=True):
            st.session_state.current_page = "library"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column layout: PDF preview and player controls
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<h2>📄 Book Preview</h2>", unsafe_allow_html=True)
        preview_path = book_data.get('preview_path')
        if preview_path and Path(preview_path).exists():
            st.image(preview_path, caption="Page 1 preview", use_column_width=True)
        else:
            st.markdown("""
            <div style="border: 3px solid #333; border-radius: 10px; padding: 40px; 
                        background-color: #F0F0F0; text-align: center; min-height: 500px;">
                <p style="font-size: 24px; color: #666;">
                    📖 Preview not available for this book yet.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<h2>🎵 Player Controls</h2>", unsafe_allow_html=True)

        # Book info
        st.markdown(f"""
        <div class="book-card">
            <p style="margin: 5px 0;">🌍 <strong>Language:</strong> {book_data['language']}</p>
            <p style="margin: 5px 0;">📅 <strong>Added:</strong> {book_data['date_added']}</p>
            <p style="margin: 5px 0;">📄 <strong>Current Page:</strong> {book_data['current_page']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Audio player using generated clips
        st.markdown("<h3>🎵 Audio Player</h3>", unsafe_allow_html=True)
        audio_clips = book_data.get('audio_clips') or []
        if not audio_clips and book_data.get('audio_path'):
            audio_clips = [book_data['audio_path']]

        if audio_clips:
            clip_labels = [f"Clip {idx + 1}: {Path(path).name}" for idx, path in enumerate(audio_clips)]
            selected_index = 0
            if len(audio_clips) > 1:
                selected_index = st.selectbox(
                    "Select audio clip",
                    options=list(range(len(audio_clips))),
                    format_func=lambda i: clip_labels[i],
                    key=f"clip_select_{book_id}"
                )
            clip_path = Path(audio_clips[selected_index])
            ext = clip_path.suffix.lower()
            mime = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            st.audio(str(clip_path), format=mime)
        else:
            st.info("🎧 Audio will appear here once generation finishes.")

        transcript_path = book_data.get('transcript_path')
        if transcript_path and Path(transcript_path).exists():
            with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
                st.download_button(
                    "⬇️ Download Transcript",
                    data=transcript_file.read(),
                    file_name=Path(transcript_path).name,
                    mime="text/plain",
                    key=f"download_transcript_{book_id}"
                )

        # Progress bar
        progress = book_data.get('progress', 0)
        st.markdown("<h3>📊 Listening Progress</h3>", unsafe_allow_html=True)
        st.progress(progress / 100)
        st.markdown(f"<p style='font-size: 20px; text-align: center;'>{progress}% Complete</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Playback settings
        st.markdown("<h3>⚙️ Playback Settings</h3>", unsafe_allow_html=True)

        # Voice selection
        voice = st.selectbox(
            "🎤 Voice",
            options=['Ana', 'Brian', 'Emma', 'James'],
            index=0,
            key="voice_select"
        )

        # Speed control
        speed = st.select_slider(
            "⚡ Playback Speed",
            options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            value=1.0,
            key="speed_select"
        )
        st.markdown(f"<p style='font-size: 20px; text-align: center;'>Current Speed: <strong>{speed}x</strong></p>", unsafe_allow_html=True)

# Settings page
def settings_page():
    """User settings page"""
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Library"):
        st.session_state.current_page = "library"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Default voice
    st.markdown("<h2>🎤 Default Voice</h2>", unsafe_allow_html=True)
    default_voice = st.selectbox(
        "Select default voice for new audiobooks",
        options=['Ana', 'Brian', 'Emma', 'James'],
        index=['Ana', 'Brian', 'Emma', 'James'].index(st.session_state.user_settings['default_voice'])
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Default speed
    st.markdown("<h2>⚡ Default Playback Speed</h2>", unsafe_allow_html=True)
    default_speed = st.select_slider(
        "Select default playback speed",
        options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        value=st.session_state.user_settings['default_speed']
    )

    st.markdown("<br>" * 2, unsafe_allow_html=True)

    # Save button
    col_save1, col_save2, col_save3 = st.columns([2, 1, 2])
    with col_save2:
        if st.button("💾 Save Settings", use_container_width=True, type="primary"):
            st.session_state.user_settings['default_voice'] = default_voice
            st.session_state.user_settings['default_speed'] = default_speed
            st.success("✅ Settings saved successfully!")

# Main app logic
def main():
    """Main application logic"""
    init_session_state()

    # Sidebar navigation (only show when authenticated)
    if st.session_state.authenticated:
        with st.sidebar:
            st.markdown("<h2 style='text-align: center;'>🎧 Navigation</h2>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("📚 Library", use_container_width=True, type="primary" if st.session_state.current_page == "library" else "secondary"):
                st.session_state.current_page = "library"
                st.rerun()

            if st.button("➕ Add Book", use_container_width=True, type="primary" if st.session_state.current_page == "add_book" else "secondary"):
                st.session_state.current_page = "add_book"
                st.rerun()

            if st.button("⚙️ Settings", use_container_width=True, type="primary" if st.session_state.current_page == "settings" else "secondary"):
                st.session_state.current_page = "settings"
                st.rerun()

            st.markdown("<br>" * 3, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"<p style='font-size: 18px; text-align: center;'>👤 {st.session_state.username}</p>", unsafe_allow_html=True)

    # Route to appropriate page
    if not st.session_state.authenticated:
        login()
    elif st.session_state.current_page == "library":
        library_page()
    elif st.session_state.current_page == "add_book":
        add_book_page()
    elif st.session_state.current_page == "player":
        player_page()
    elif st.session_state.current_page == "settings":
        settings_page()

if __name__ == "__main__":
    main()
