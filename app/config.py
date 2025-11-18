"""
Configuration file for Audiobook Creator Application
Centralized settings and constants
"""

# Application Settings
APP_NAME = "booq - Audiobook Creator"
APP_VERSION = "1.0.0"
APP_ICON = "🎧"

# Directory Configuration
DATA_DIR = "data"
PDF_DIR = "data/pdfs"
AUDIO_DIR = "data/audio"
CACHE_DIR = "data/audio_cache"
LIBRARY_FILE = "data/library.json"

# Supported Languages
LANGUAGES = [
    'English',
    'Spanish',
    'French',
    'German',
    'Italian',
    'Portuguese',
    'Korean',
    'Japanese'
]

# Language Code Mapping (for TTS)
LANGUAGE_CODES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Korean': 'ko',
    'Japanese': 'ja'
}

# Available Voices
VOICES = [
    'Ana',
    'Brian',
    'Emma',
    'James'
]

# Playback Speed Options
PLAYBACK_SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Default Settings
DEFAULT_LANGUAGE = 'English'
DEFAULT_VOICE = 'Ana'
DEFAULT_SPEED = 1.0
DEFAULT_THEME = 'light'

# Accessibility Settings
BASE_FONT_SIZE = 18  # pixels
HEADING_1_SIZE = 48  # pixels
HEADING_2_SIZE = 36  # pixels
HEADING_3_SIZE = 28  # pixels
BUTTON_FONT_SIZE = 24  # pixels
BUTTON_MIN_HEIGHT = 60  # pixels

# File Upload Settings
MAX_FILE_SIZE_MB = 200  # Maximum PDF file size in MB
ALLOWED_FILE_TYPES = ['pdf']

# Audio Generation Settings
WORDS_PER_MINUTE = 150  # Average reading speed for duration estimation
CHARACTERS_PER_WORD = 5  # Average characters per word

# Cache Settings
ENABLE_CACHE = True
CACHE_EXPIRY_DAYS = 30  # Days before cached audio is deleted

# Session Settings
SESSION_TIMEOUT_MINUTES = 60  # Inactivity timeout

# UI Colors (High Contrast for Accessibility)
COLORS = {
    'background': '#FFFFFF',
    'text': '#000000',
    'primary': '#0066CC',
    'secondary': '#333333',
    'accent': '#FF6600',
    'success': '#00AA00',
    'warning': '#FFAA00',
    'error': '#CC0000',
    'card_bg': '#F8F8F8',
    'border': '#333333'
}

# Progress Bar Settings
PROGRESS_STEPS = {
    'pdf_extraction': 20,
    'text_processing': 30,
    'audio_generation': 70,
    'saving': 90,
    'complete': 100
}

# Logging Configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model Configuration (Update with your model specifics)
MODEL_CONFIG = {
    'model_path': None,  # Path to your pre-trained model
    'model_type': 'custom',  # 'custom', 'gtts', 'transformers', etc.
    'batch_size': 32,
    'max_length': 1000  # Maximum characters per audio chunk
}

# Feature Flags
FEATURES = {
    'user_authentication': True,
    'search_functionality': True,
    'delete_confirmation': True,
    'audio_caching': True,
    'progress_tracking': True,
    'multi_language': True,
    'voice_selection': True,
    'speed_control': True,
    'pdf_preview': True,
    'settings_page': True
}

# Error Messages
ERROR_MESSAGES = {
    'pdf_upload_failed': '⚠️ Failed to upload PDF file. Please try again.',
    'pdf_processing_failed': '⚠️ Error processing PDF. Please ensure the file is valid.',
    'audio_generation_failed': '⚠️ Error generating audio. Please try again.',
    'book_not_found': '⚠️ Book not found in library.',
    'authentication_failed': '⚠️ Invalid username or password.',
    'file_too_large': '⚠️ File is too large. Maximum size is {max_size}MB.',
    'invalid_file_type': '⚠️ Invalid file type. Please upload a PDF file.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'audiobook_created': '✅ Audiobook created successfully!',
    'book_deleted': '✅ Book deleted successfully!',
    'settings_saved': '✅ Settings saved successfully!',
    'login_successful': '✅ Login successful!'
}

# Info Messages
INFO_MESSAGES = {
    'empty_library': '📖 Your library is empty. Click "Add New Book" to get started!',
    'creating_audiobook': '🔄 Creating your audiobook... This may take a few minutes.',
    'processing_pdf': '📖 Reading PDF content...',
    'generating_audio': '🤖 Generating audio with AI...',
    'saving_audiobook': '💾 Saving audiobook...'
}
