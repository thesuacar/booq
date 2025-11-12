# --- Added by assistant: PDF embed helper ---
def show_pdf_inline(pdf_path: str, height: int = 600):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        st.warning("PDF not found.")
        return
    import base64 as _b64
    b64 = _b64.b64encode(pdf_file.read_bytes()).decode("utf-8")
    html(f"""
        <iframe
          src="data:application/pdf;base64,{b64}"
          style="width:100%; height:{height}px; border:none;"
        ></iframe>
    """, height=height)


"""
Audiobook Creator Application - Main UI
Built with Streamlit for accessibility (VPD-focused)
This application helps create AI-generated audiobooks from PDF files
"""

import streamlit as st
import base64
from streamlit.components.v1 import html

import os
import json
from pathlib import Path
import time
from datetime import datetime

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

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
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
    library_path = Path("data/library.json")
    if library_path.exists():
        with open(library_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_library():
    """Save library to JSON file"""
    Path("data").mkdir(exist_ok=True)
    with open("data/library.json", 'w', encoding='utf-8') as f:
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
                # Delete book files
                if 'audio_path' in library[book_id]:
                    audio_path = Path(library[book_id]['audio_path'])
                    if audio_path.exists():
                        os.remove(audio_path)

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
    language = st.selectbox(
        "Choose language for audiobook",
        options=['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Korean', 'Japanese'],
        index=0
    )

    st.markdown("<br>" * 2, unsafe_allow_html=True)

    # Create audiobook button
    col_create1, col_create2, col_create3 = st.columns([1, 1, 1])
    with col_create2:
        create_button = st.button("🎙️ Create Audiobook", use_container_width=True, type="primary", disabled=not uploaded_file)

    if create_button and uploaded_file:
        # Save PDF
        Path("data/pdfs").mkdir(parents=True, exist_ok=True)
        pdf_filename = f"{int(time.time())}_{uploaded_file.name}"
        pdf_path = Path("data/pdfs") / pdf_filename

        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Show processing message
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔄 Creating your audiobook... This may take a few minutes.")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate processing (in real app, this would call the AI model)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("📖 Reading PDF content...")
            elif i < 60:
                status_text.text("🤖 Generating audio with AI...")
            else:
                status_text.text("💾 Saving audiobook...")

        # Create book entry
        book_id = f"book_{int(time.time())}"
        book_title = uploaded_file.name.replace('.pdf', '')

        st.session_state.library[book_id] = {
            'title': book_title,
            'language': language,
            'date_added': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'pdf_path': str(pdf_path),
            'audio_path': f"data/audio/{book_id}.mp3",  # Would be generated by model
            'status': 'Ready',
            'progress': 0,
            'duration': '00:00:00',  # Would be calculated
            'current_page': 1,
            'total_pages': 0  # Would be extracted from PDF
        }

        save_library()

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

        # PDF preview placeholder
        st.markdown("""
        <div style="border: 3px solid #333; border-radius: 10px; padding: 40px; 
                    background-color: #F0F0F0; text-align: center; min-height: 500px;">
            <p style="font-size: 24px; color: #666;">
                📖 PDF Preview<br><br>
                Page 1 of the book would be displayed here
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

        # Audio player placeholder (would use st.audio with actual file)
        st.markdown("<h3>🎵 Audio Player</h3>", unsafe_allow_html=True)
        st.info("🎧 Audio player would appear here. In production, this would play the generated audiobook.")

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


# --- Added by assistant: Player screen ---
def screen_player():
    book_id = st.session_state.get("selected_book")
    if not book_id or book_id not in st.session_state.library:
        st.info("No book selected. Go to Library to pick one.")
        if st.button("📚 Go to Library"):
            st.session_state.current_page = "library"
            st.rerun()
        return

    book = st.session_state.library[book_id]
    title = book.get("title", "Untitled")
    audio_path = book.get("audio_path")
    pdf_path = book.get("pdf_path")

    st.header(f"🎧 {title}")
    st.write("Use the controls to play or pause the generated audio.")

    if audio_path and Path(audio_path).exists():
        st.audio(str(audio_path))
    else:
        st.warning("Audio file not found.")

    st.divider()
    st.subheader("📄 Book PDF")
    if pdf_path:
        show_pdf_inline(str(pdf_path), height=650)
    else:
        st.info("No PDF available for this book.")

    st.divider()
    if st.button("📚 Back to Library"):
        st.session_state.current_page = "library"
        st.rerun()



# --- Added by assistant: Library screen ---
def screen_library():
    st.header("📚 Your Library")

    if not st.session_state.library:
        st.info("Your library is empty. Create an audiobook to get started!")
        return

    for bid, meta in st.session_state.library.items():
        with st.container(border=True):
            st.write(f"**{meta.get('title','Untitled')}**")
            cols = st.columns([1,1,1])
            with cols[0]:
                if st.button("▶️ Play", key=f"play_{bid}"):
                    st.session_state.selected_book = bid
                    st.session_state.current_page = "player"
                    st.rerun()
            with cols[1]:
                ap = meta.get("audio_path")
                if ap and Path(ap).exists():
                    try:
                        data = Path(ap).read_bytes()
                        st.download_button("⬇️ Download Audio", data=data, file_name=Path(ap).name, key=f"dl_audio_{bid}")
                    except Exception:
                        pass
            with cols[2]:
                pp = meta.get("pdf_path")
                if pp and Path(pp).exists():
                    try:
                        data = Path(pp).read_bytes()
                        st.download_button("⬇️ Download PDF", data=data, file_name=Path(pp).name, key=f"dl_pdf_{bid}")
                    except Exception:
                        pass



# --- Added by assistant: Simple router ---
_page = st.session_state.get("current_page", "home")
if _page == "player":
    screen_player()
elif _page == "library":
    screen_library()
else:
    pass  # your existing home/create UI handles default flow
