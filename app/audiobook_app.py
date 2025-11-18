"""
Audiobook Creator Application - Main UI (Streamlit)
Distributed architecture: talks to orchestrator_service via HTTP.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import requests
import streamlit as st
from streamlit.components.v1 import html

# ---------- CONFIG ----------

ORCHESTRATOR_URL = "http://localhost:8001"  # FastAPI orchestrator base URL

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="booq - Audiobook Creator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- ACCESSIBLE CSS ----------

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
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
    .stButton > button {
        font-size: 24px !important;
        padding: 20px 40px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        min-height: 60px !important;
    }
    .stTextInput > div > div > input {
        font-size: 22px !important;
        padding: 15px !important;
        min-height: 50px !important;
    }
    .stSelectbox > div > div > select {
        font-size: 22px !important;
        padding: 15px !important;
    }
    .stFileUploader > div {
        font-size: 20px !important;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    .book-card {
        border: 3px solid #333;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        background-color: #F8F8F8;
        font-size: 20px;
    }
    .stProgress > div > div {
        height: 30px !important;
    }
    audio {
        width: 100% !important;
        height: 60px !important;
    }
    .stAlert {
        font-size: 20px !important;
        padding: 20px !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 24px !important;
        padding: 15px 30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HELPERS ----------


def show_pdf_inline(pdf_path: str, height: int = 600):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        st.warning("PDF not found.")
        return
    import base64 as _b64

    b64 = _b64.b64encode(pdf_file.read_bytes()).decode("utf-8")
    html(
        f"""
        <iframe
          src="data:application/pdf;base64,{b64}"
          style="width:100%; height:{height}px; border:none;"
        ></iframe>
    """,
        height=height,
    )


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "library" not in st.session_state:
        st.session_state.library = load_library()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "login"
    if "selected_book" not in st.session_state:
        st.session_state.selected_book = None
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {
            "default_voice": "Ana",
            "default_speed": 1.0,
            "theme": "light",
        }


def load_library() -> Dict[str, Any]:
    library_path = Path("data/library.json")
    if library_path.exists():
        with library_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_library():
    Path("data").mkdir(exist_ok=True)
    with Path("data/library.json").open("w", encoding="utf-8") as f:
        json.dump(st.session_state.library, f, indent=2, ensure_ascii=False)


# ---------- AUTH ----------


def login():
    st.markdown(
        "<h1 style='text-align: center;'>🎧 booq - Audiobook Creator</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center;'>Login to Access Your Library</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        username = st.text_input("👤 Username", key="login_username", placeholder="Enter your username")
        password = st.text_input(
            "🔒 Password", type="password", key="login_password", placeholder="Enter your password"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Login", use_container_width=True):
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.current_page = "library"
                    st.rerun()
                else:
                    st.error("⚠️ Please enter both username and password")


def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.current_page = "login"
    st.rerun()


# ---------- PAGES ----------


def library_page():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1>📚 My Audiobook Library</h1>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size: 22px;'>Welcome back, <strong>{st.session_state.username}</strong>!</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    search_query = st.text_input("🔍 Search books by title", placeholder="Type to search...")

    col_add1, col_add2, col_add3 = st.columns([2, 1, 2])
    with col_add2:
        if st.button("➕ Add New Book", use_container_width=True, type="primary"):
            st.session_state.current_page = "add_book"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    library = st.session_state.library
    if not library:
        st.info("📖 Your library is empty. Click 'Add New Book' to get started!")
        return

    filtered = {
        book_id: data
        for book_id, data in library.items()
        if search_query.lower() in data["title"].lower()
    }

    if not filtered:
        st.warning(f"No books found matching '{search_query}'")
        return

    st.markdown(f"<h3>Found {len(filtered)} book(s)</h3>", unsafe_allow_html=True)

    for book_id, book_data in filtered.items():
        with st.container():
            col_book1, col_book2 = st.columns([4, 1])

            with col_book1:
                st.markdown(
                    f"""
                    <div class="book-card">
                        <h2 style="margin: 0;">{book_data['title']}</h2>
                        <p style="font-size: 20px; margin: 10px 0;">
                            🌍 Language: <strong>{book_data['language']}</strong><br>
                            📅 Added: {book_data['date_added']}<br>
                            🎵 Status: <strong>{book_data['status']}</strong><br>
                            ⏱ Duration: <strong>{book_data.get('duration','00:00:00')}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_book2:
                st.markdown("<br>" * 2, unsafe_allow_html=True)
                if st.button("▶️ Play", key=f"play_{book_id}", use_container_width=True):
                    st.session_state.selected_book = book_id
                    st.session_state.current_page = "player"
                    st.rerun()

                if st.button("🗑️ Delete", key=f"delete_{book_id}", use_container_width=True):
                    st.session_state.book_to_delete = book_id
                    st.session_state.show_delete_confirm = True

    if "show_delete_confirm" in st.session_state and st.session_state.show_delete_confirm:
        bid = st.session_state.book_to_delete
        book_title = library[bid]["title"]

        st.markdown("---")
        st.warning(f"⚠️ Are you sure you want to delete '{book_title}'?")

        col_del1, col_del2, col_del3 = st.columns([2, 1, 1])
        with col_del2:
            if st.button("✅ Yes, Delete", use_container_width=True):
                # Delete files if they exist
                audio_path = library[bid].get("audio_path")
                if audio_path and Path(audio_path).exists():
                    os.remove(audio_path)
                pdf_path = library[bid].get("pdf_path")
                if pdf_path and Path(pdf_path).exists():
                    os.remove(pdf_path)

                del st.session_state.library[bid]
                save_library()
                st.session_state.show_delete_confirm = False
                st.rerun()
        with col_del3:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_delete_confirm = False
                st.rerun()


def add_book_page():
    st.markdown("<h1>➕ Add New Audiobook</h1>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Library"):
        st.session_state.current_page = "library"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>1️⃣ Upload Your Book (PDF)</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], help="Upload a PDF book to convert to audiobook"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>2️⃣ Select Audiobook Language</h2>", unsafe_allow_html=True)
    language = st.selectbox(
        "Choose language for audiobook",
        options=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Korean", "Japanese"],
        index=0,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_create1, col_create2, col_create3 = st.columns([1, 1, 1])
    with col_create2:
        create_button = st.button(
            "🎙️ Create Audiobook",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_file,
        )

    if create_button and uploaded_file:
        pdf_dir = Path("data/pdfs")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        pdf_filename = f"{timestamp}_{uploaded_file.name}"
        pdf_path = pdf_dir / pdf_filename

        with pdf_path.open("wb") as f:
            f.write(uploaded_file.getvalue())

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔄 Sending job to orchestrator...")

        progress_bar = st.progress(10)
        status_text = st.empty()

        book_id = f"book_{timestamp}"
        book_title = uploaded_file.name.replace(".pdf", "")

        # Map UI language to code (simple)
        lang_code = "en"  # TODO: map properly if needed

        voice = st.session_state.user_settings.get("default_voice", "Ana")
        speed = st.session_state.user_settings.get("default_speed", 1.0)

        payload = {
            "book_id": book_id,
            "pdf_path": str(pdf_path),
            "language": lang_code,
            "voice": voice,
            "speed": float(speed),
        }

        try:
            status_text.text("📡 Contacting orchestrator...")
            resp = requests.post(f"{ORCHESTRATOR_URL}/audiobooks/create", json=payload, timeout=600)
            progress_bar.progress(80)
        except Exception as exc:
            st.error(f"❌ Failed to contact orchestrator: {exc}")
            return

        if resp.status_code != 200:
            st.error(f"❌ Orchestrator error: HTTP {resp.status_code}")
            return

        result = resp.json()
        if not result.get("success"):
            st.error(f"❌ Failed to create audiobook: {result.get('error', 'Unknown error')}")
            return

        audio_path = result["audio_path"]
        text_path = result["text_path"]
        duration = result.get("duration", "00:00:00")
        total_pages = result.get("total_pages", 0)

        st.session_state.library[book_id] = {
            "title": book_title,
            "language": language,
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pdf_path": str(pdf_path),
            "audio_path": audio_path,
            "text_path": text_path,
            "status": "Ready",
            "progress": 0,
            "duration": duration,
            "current_page": 1,
            "total_pages": total_pages,
        }
        save_library()

        progress_bar.progress(100)
        status_text.text("✅ Audiobook created!")

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


def player_page():
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

    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown(f"<h1>🎧 {book_data['title']}</h1>", unsafe_allow_html=True)
    with col_header2:
        if st.button("⬅️ Library", use_container_width=True):
            st.session_state.current_page = "library"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<h2>📄 Book Preview</h2>", unsafe_allow_html=True)
        pdf_path = book_data.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            show_pdf_inline(pdf_path, height=650)
        else:
            st.info("No PDF preview available.")

    with col_right:
        st.markdown("<h2>🎵 Player Controls</h2>", unsafe_allow_html=True)

        duration = book_data.get("duration", "00:00:00")
        total_pages = book_data.get("total_pages", 0)
        st.markdown(
            f"""
            <div class="book-card">
                <p style="margin: 5px 0;">🌍 <strong>Language:</strong> {book_data['language']}</p>
                <p style="margin: 5px 0;">📅 <strong>Added:</strong> {book_data['date_added']}</p>
                <p style="margin: 5px 0;">📄 <strong>Pages:</strong> {total_pages}</p>
                <p style="margin: 5px 0;">⏱ <strong>Duration (est):</strong> {duration}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<h3>🎵 Audio Player</h3>", unsafe_allow_html=True)
        audio_path = book_data.get("audio_path")
        if audio_path and Path(audio_path).exists():
            with Path(audio_path).open("rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.warning("⚠️ Audio file not found. Please recreate this audiobook.")

        progress = book_data.get("progress", 0)
        st.markdown("<h3>📊 Listening Progress</h3>", unsafe_allow_html=True)
        st.progress(progress / 100 if progress <= 100 else 1.0)
        st.markdown(
            f"<p style='font-size: 20px; text-align: center;'>{progress}% Complete</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<h3>⚙️ Playback Settings</h3>", unsafe_allow_html=True)
        voice = st.selectbox(
            "🎤 Voice",
            options=["Ana", "Brian", "Emma", "James"],
            index=0,
            key="voice_select",
        )
        speed = st.select_slider(
            "⚡ Playback Speed",
            options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            value=1.0,
            key="speed_select",
        )
        st.markdown(
            f"<p style='font-size: 20px; text-align: center;'>Current Speed: <strong>{speed}x</strong></p>",
            unsafe_allow_html=True,
        )


def settings_page():
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)

    if st.button("⬅️ Back to Library"):
        st.session_state.current_page = "library"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>🎤 Default Voice</h2>", unsafe_allow_html=True)
    default_voice = st.selectbox(
        "Select default voice for new audiobooks",
        options=["Ana", "Brian", "Emma", "James"],
        index=["Ana", "Brian", "Emma", "James"].index(
            st.session_state.user_settings["default_voice"]
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h2>⚡ Default Playback Speed</h2>", unsafe_allow_html=True)
    default_speed = st.select_slider(
        "Select default playback speed",
        options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        value=st.session_state.user_settings["default_speed"],
    )

    st.markdown("<br>" * 2, unsafe_allow_html=True)

    col_save1, col_save2, col_save3 = st.columns([2, 1, 2])
    with col_save2:
        if st.button("💾 Save Settings", use_container_width=True, type="primary"):
            st.session_state.user_settings["default_voice"] = default_voice
            st.session_state.user_settings["default_speed"] = default_speed
            st.success("✅ Settings saved successfully!")


# ---------- MAIN ----------


def main():
    init_session_state()

    if st.session_state.authenticated:
        with st.sidebar:
            st.markdown("<h2 style='text-align: center;'>🎧 Navigation</h2>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "📚 Library",
                use_container_width=True,
                type="primary" if st.session_state.current_page == "library" else "secondary",
            ):
                st.session_state.current_page = "library"
                st.rerun()

            if st.button(
                "➕ Add Book",
                use_container_width=True,
                type="primary" if st.session_state.current_page == "add_book" else "secondary",
            ):
                st.session_state.current_page = "add_book"
                st.rerun()

            if st.button(
                "⚙️ Settings",
                use_container_width=True,
                type="primary" if st.session_state.current_page == "settings" else "secondary",
            ):
                st.session_state.current_page = "settings"
                st.rerun()

            st.markdown("<br>" * 3, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(
                f"<p style='font-size: 18px; text-align: center;'>👤 {st.session_state.username}</p>",
                unsafe_allow_html=True,
            )

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
