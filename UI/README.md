# Audiobook Creator Application

## Overview
This is an accessible audiobook creation application built with Streamlit, designed specifically for vision-impaired users (particularly children with Visual Processing Disorder - VPD). The application converts PDF books into AI-generated audiobooks with multi-language support.

## Project Structure

```
audiobook-app/
├── audiobook_app.py              # Main Streamlit UI application
├── audiobook_orchestrator.ipynb  # AI model integration orchestrator
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── config.py                     # Configuration file
└── data/                         # Data directory
    ├── pdfs/                     # Uploaded PDF files
    ├── audio/                    # Generated audiobook files
    ├── audio_cache/              # Cached audio segments
    └── library.json              # Library database
```

## Features

### 1. User Authentication
- Simple login system without complex role management
- Session-based authentication using Streamlit session state
- Persistent login across page navigation

### 2. Library Management
- View all audiobooks in a clean, accessible interface
- Search functionality to find books by title
- Delete books with confirmation dialog
- Large fonts and high-contrast design for accessibility

### 3. Add New Books
- Upload PDF files
- Select audiobook language (8+ languages supported)
- AI-powered text-to-speech conversion
- Progress tracking during generation

### 4. Audio Player
- Display PDF preview (first page)
- Audio playback controls
- Progress tracking that persists across sessions
- Adjustable playback speed (0.5x to 2.0x)
- Multiple voice options
- Settings saved per user

### 5. Accessibility Features
- **Extra large fonts** (18px base, up to 48px for headers)
- **High contrast** color scheme
- **Large buttons** (minimum 60px height)
- **Clear spacing** between elements
- **Simple navigation** with big, clear icons
- Designed following WCAG accessibility guidelines

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create required directories**
```bash
mkdir -p data/pdfs data/audio data/audio_cache
```

4. **Run the application**
```bash
streamlit run audiobook_app.py
```

5. **Access the application**
   - Open your browser
   - Navigate to `http://localhost:8501`
   - Login with any username/password (simple authentication)

## Required Libraries

```
streamlit>=1.28.0
PyPDF2>=3.0.0
gtts>=2.4.0
pydub>=0.25.1
langdetect>=1.0.9
```

Optional (for advanced AI models):
```
transformers>=4.30.0
torch>=2.0.0
```

## Usage Guide

### For Users

#### 1. Login
- Enter any username and password
- Click "Login" to access the application

#### 2. Add a New Book
- Click "Add New Book" button
- Upload a PDF file
- Select the desired language
- Click "Create Audiobook"
- Wait for processing to complete
- Click "Play Now" or return to library

#### 3. Play an Audiobook
- From the library, click "Play" on any book
- Use the audio player controls
- Adjust speed and voice settings as needed
- Progress is automatically saved

#### 4. Manage Library
- Search for books using the search bar
- Delete books by clicking "Delete" and confirming

### For Developers

#### Integrating the AI Model

The application is designed to work with your pre-trained TTS model. Follow these steps:

1. **Open `audiobook_orchestrator.ipynb`**

2. **Modify the `AudioGenerator` class**:
```python
def load_model(self):
    # Replace this with your model loading code
    from your_model_package import YourTTSModel
    self.model = YourTTSModel.load(self.model_path)
```

3. **Update the `generate_audio_from_text` method**:
```python
def generate_audio_from_text(self, text, language, voice, speed):
    # Use your model for inference
    audio_data = self.model.synthesize(
        text=text,
        language=language,
        voice=voice,
        speed=speed
    )
    return audio_data
```

4. **Import the orchestrator in `audiobook_app.py`**:
```python
# Add at the top of audiobook_app.py
from audiobook_orchestrator import AudiobookManager

# Initialize in the app
@st.cache_resource
def get_audiobook_manager():
    return AudiobookManager(model_path="path/to/your/model")
```

#### Audio Caching System

The orchestrator includes a built-in caching mechanism to avoid regenerating the same audio:

- Audio segments are cached based on: text content, language, voice, and speed
- Cache files are stored in `data/audio_cache/`
- This significantly improves performance for repeated content

#### Data Persistence

The application uses multiple persistence mechanisms:

1. **Session State**: For temporary data within a user session
2. **JSON Database**: `data/library.json` stores all book metadata
3. **File System**: PDFs and audio files are stored in `data/` directory

## Configuration

### Customizing Settings

Edit `config.py` to customize:

```python
# Available languages
LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Korean', 'Japanese']

# Available voices
VOICES = ['Ana', 'Brian', 'Emma', 'James']

# Playback speeds
SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Default settings
DEFAULT_VOICE = 'Ana'
DEFAULT_SPEED = 1.0
DEFAULT_LANGUAGE = 'English'
```

### Accessibility Settings

The CSS styling in `audiobook_app.py` can be customized:

```css
/* Base font size - increase for larger text */
html, body, [class*="css"] {
    font-size: 18px !important;
}

/* Header sizes */
h1 { font-size: 48px !important; }
h2 { font-size: 36px !important; }
h3 { font-size: 28px !important; }

/* Button sizes */
.stButton > button {
    font-size: 24px !important;
    min-height: 60px !important;
}
```

## Architecture

### Component Overview

1. **audiobook_app.py** (Frontend)
   - Streamlit-based user interface
   - Session management
   - Navigation and routing
   - User interactions

2. **audiobook_orchestrator.ipynb** (Backend)
   - PDF text extraction
   - AI model integration
   - Audio generation and caching
   - Progress tracking

3. **Data Layer**
   - JSON database for metadata
   - File system for PDFs and audio
   - Cache system for optimization

### Data Flow

```
User uploads PDF
    ↓
PDF saved to data/pdfs/
    ↓
Orchestrator extracts text (PyPDF2)
    ↓
Text sent to AI model
    ↓
Audio generated and cached
    ↓
Audio saved to data/audio/
    ↓
Metadata stored in library.json
    ↓
User can play audiobook
```

## Troubleshooting

### Common Issues

**1. Audio not playing**
- Check that audio files exist in `data/audio/`
- Verify browser supports MP3 playback
- Check browser console for errors

**2. PDF upload fails**
- Ensure `data/pdfs/` directory exists
- Check file size limits (Streamlit default: 200MB)
- Verify PDF is not corrupted

**3. Session state lost on refresh**
- This is normal Streamlit behavior
- User needs to log in again after refresh
- Book library data persists in `library.json`

**4. Slow performance**
- Enable audio caching (default)
- Consider preprocessing large PDFs
- Use smaller chunks for audio generation

### Debug Mode

To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
- [ ] Bookmark specific pages/timestamps
- [ ] Export/import library
- [ ] Batch upload multiple PDFs
- [ ] Advanced user management with roles
- [ ] Cloud storage integration
- [ ] Mobile app version
- [ ] Voice commands for navigation
- [ ] Multi-user support with individual libraries
- [ ] Progress synchronization across devices
- [ ] Image description using vision AI

### Model Improvements
- [ ] Better voice quality with neural TTS
- [ ] Emotion and emphasis in narration
- [ ] Multiple speaker support
- [ ] Real-time speed adjustment without regeneration
- [ ] Background music/sound effects

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with accessibility tools
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write accessible code (consider screen readers)

## License

This project is created for educational purposes as part of a VPD assistance initiative.

## Support

For issues or questions:
- Check the documentation above
- Review the code comments
- Test with the provided examples
- Contact the development team

## Acknowledgments

- Designed for children with Visual Processing Disorder (VPD)
- Built with Streamlit framework
- Follows WCAG accessibility guidelines
- Inspired by inclusive education principles

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Authors**: Development Team
