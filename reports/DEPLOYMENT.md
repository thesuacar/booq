# Deployment and Usage Guide

## Quick Start Guide

### For End Users

#### Step 1: Initial Setup
1. Ensure you have Python 3.8+ installed on your computer
2. Download the application files
3. Open a terminal/command prompt
4. Navigate to the application folder

#### Step 2: Installation
Run the following commands:

```bash
# Install required packages
pip install -r requirements.txt

# Create data directories
mkdir data
mkdir data/pdfs
mkdir data/audio
mkdir data/audio_cache
```

#### Step 3: Launch the Application
```bash
streamlit run audiobook_app.py
```

The application will open in your default web browser at `http://localhost:8501`

#### Step 4: First Use
1. **Login Screen**: Enter any username and password (simple authentication)
2. **Library Page**: Initially empty, you'll see a welcome message
3. **Add Your First Book**:
   - Click "➕ Add New Book"
   - Upload a PDF file
   - Select your preferred language
   - Click "🎙️ Create Audiobook"
   - Wait for processing (progress bar will show status)
4. **Play Your Audiobook**:
   - Return to library or click "▶️ Play Now"
   - Adjust speed and voice settings
   - Your progress will be saved automatically

---

## For Developers

### Development Environment Setup

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for version control)
- Code editor (VS Code recommended)

#### Setup Steps

1. **Clone/Download the Project**
```bash
git clone <repository-url>
cd audiobook-app
```

2. **Create Virtual Environment** (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Directory Structure**
```
audiobook-app/
├── audiobook_app.py              # Main Streamlit UI
├── audiobook_orchestrator.ipynb  # AI integration notebook
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── DEPLOYMENT.md                 # This file
└── data/                         # Data directory (created automatically)
    ├── pdfs/                     # Uploaded PDFs
    ├── audio/                    # Generated audiobooks
    ├── audio_cache/              # Cached audio segments
    └── library.json              # Library database
```

### Integrating Your AI Model

#### Option 1: Using the Orchestrator Notebook

1. **Open `audiobook_orchestrator.ipynb`**

2. **Locate the `AudioGenerator` class**

3. **Modify the `load_model()` method**:
```python
def load_model(self):
    """Load your trained TTS model"""
    import your_model_package

    self.model = your_model_package.load_model(
        model_path=self.model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    logger.info("Model loaded successfully")
```

4. **Update `generate_audio_from_text()` method**:
```python
def generate_audio_from_text(self, text, language, voice, speed):
    """Generate audio using your model"""

    # Preprocess text if needed
    processed_text = self.preprocess_text(text)

    # Generate audio with your model
    audio_data = self.model.synthesize(
        text=processed_text,
        language=language,
        voice=voice,
        speed=speed
    )

    # Post-process if needed
    audio_data = self.postprocess_audio(audio_data, speed)

    return audio_data
```

#### Option 2: Creating a Standalone Module

Create a file `tts_model.py`:

```python
"""
Custom TTS Model Integration
"""

class CustomTTSModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load your model"""
        # Your model loading code here
        pass

    def synthesize(self, text, language='en', voice='default', speed=1.0):
        """Generate speech from text"""
        # Your synthesis code here
        pass
```

Then import it in the orchestrator:

```python
from tts_model import CustomTTSModel

class AudioGenerator:
    def __init__(self, model_path):
        self.tts_model = CustomTTSModel(model_path)
        self.tts_model.load()
```

### Database Schema

The application uses a JSON file (`data/library.json`) to store book metadata:

```json
{
  "book_123456789": {
    "title": "Sample Book Title",
    "language": "English",
    "date_added": "2025-11-09 21:30",
    "pdf_path": "data/pdfs/123456789_sample.pdf",
    "audio_path": "data/audio/book_123456789.mp3",
    "status": "Ready",
    "progress": 0,
    "duration": "01:23:45",
    "current_page": 1,
    "total_pages": 150
  }
}
```

### Session State Variables

The application uses Streamlit session state to manage:

```python
st.session_state.authenticated      # Boolean: User login status
st.session_state.username          # String: Current username
st.session_state.library           # Dict: All books
st.session_state.current_page      # String: Current page/view
st.session_state.selected_book     # String: Currently playing book ID
st.session_state.user_settings     # Dict: User preferences
```

### API Reference

#### AudiobookManager Class

```python
class AudiobookManager:
    """Main orchestrator for audiobook creation"""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the manager"""

    def create_audiobook(
        self,
        pdf_path: str,
        book_id: str,
        language: str = 'English',
        voice: str = 'default',
        speed: float = 1.0,
        progress_callback = None
    ) -> Dict:
        """
        Create complete audiobook from PDF

        Args:
            pdf_path: Path to PDF file
            book_id: Unique identifier
            language: Target language
            voice: Voice identifier
            speed: Playback speed multiplier
            progress_callback: Function(percent, message)

        Returns:
            Dict with 'success', 'audio_path', 'total_pages', 'duration'
        """
```

#### PDFProcessor Class

```python
class PDFProcessor:
    """Handles PDF processing"""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
        """Extract text page by page"""

    @staticmethod
    def get_pdf_metadata(pdf_path: str) -> Dict:
        """Get PDF metadata"""
```

### Testing

#### Unit Tests

Create `tests/test_orchestrator.py`:

```python
import unittest
from audiobook_orchestrator import AudiobookManager, PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    def test_extract_text(self):
        processor = PDFProcessor()
        # Add test PDF
        texts = processor.extract_text_from_pdf('tests/sample.pdf')
        self.assertIsInstance(texts, dict)
        self.assertGreater(len(texts), 0)

class TestAudiobookManager(unittest.TestCase):
    def test_create_audiobook(self):
        manager = AudiobookManager()
        result = manager.create_audiobook(
            pdf_path='tests/sample.pdf',
            book_id='test_123'
        )
        self.assertTrue(result['success'])

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest tests/
```

### Deployment Options

#### Option 1: Local Deployment (Development)

```bash
streamlit run audiobook_app.py
```

**Pros**: Fast, easy for development  
**Cons**: Only accessible on local machine

#### Option 2: Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Deploy

**Pros**: Free, easy deployment  
**Cons**: Limited resources, public access

#### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "audiobook_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t audiobook-app .
docker run -p 8501:8501 audiobook-app
```

**Pros**: Consistent environment, easy scaling  
**Cons**: Requires Docker knowledge

#### Option 4: Cloud Platform (AWS/GCP/Azure)

Deploy to cloud VM:

```bash
# On cloud instance
git clone <repo>
cd audiobook-app
pip install -r requirements.txt
streamlit run audiobook_app.py --server.port=80
```

**Pros**: Full control, scalable  
**Cons**: Cost, requires setup

### Performance Optimization

#### 1. Caching Strategy

The application implements caching at multiple levels:

- **Audio Cache**: Stores generated audio segments
- **Session Cache**: Uses Streamlit's `@st.cache_data` and `@st.cache_resource`
- **PDF Cache**: Stores extracted text

#### 2. Chunking Large PDFs

For books with many pages:

```python
def process_large_pdf(pdf_path, chunk_size=10):
    """Process PDF in chunks"""
    processor = PDFProcessor()
    page_texts = processor.extract_text_from_pdf(pdf_path)

    chunks = []
    current_chunk = []

    for page_num, text in page_texts.items():
        current_chunk.append(text)
        if len(current_chunk) >= chunk_size:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
```

#### 3. Async Audio Generation

For better responsiveness:

```python
import asyncio

async def generate_audio_async(text, language):
    """Generate audio asynchronously"""
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(
        None, 
        audio_generator.generate_audio_from_text,
        text,
        language
    )
    return audio
```

### Monitoring and Logging

#### Setup Logging

Add to `audiobook_app.py`:

```python
import logging
from datetime import datetime

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

#### Track Usage

```python
def log_user_action(action, details):
    """Log user actions for analytics"""
    logger.info(f"User: {st.session_state.username} | Action: {action} | Details: {details}")

# Example usage
log_user_action('create_audiobook', {'language': language, 'pages': total_pages})
```

### Security Considerations

#### 1. Authentication Enhancement

Replace simple auth with proper authentication:

```python
import bcrypt
import json

def verify_password(username, password):
    """Verify password against stored hash"""
    with open('users.json', 'r') as f:
        users = json.load(f)

    if username in users:
        stored_hash = users[username]['password_hash']
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    return False
```

#### 2. File Upload Validation

```python
def validate_pdf(uploaded_file):
    """Validate uploaded PDF file"""
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large: {uploaded_file.size} bytes")

    # Check file type
    if not uploaded_file.name.endswith('.pdf'):
        raise ValueError("Invalid file type")

    # Verify it's actually a PDF
    try:
        PyPDF2.PdfReader(uploaded_file)
    except Exception:
        raise ValueError("Invalid PDF file")
```

#### 3. Sanitize User Input

```python
import re

def sanitize_filename(filename):
    """Remove potentially dangerous characters"""
    # Remove any path separators
    filename = filename.replace('/', '').replace('\\', '')
    # Keep only alphanumeric, dash, underscore, dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename
```

### Troubleshooting

#### Common Issues and Solutions

1. **Port Already in Use**
```bash
streamlit run audiobook_app.py --server.port=8502
```

2. **Module Not Found**
```bash
pip install -r requirements.txt --upgrade
```

3. **PDF Extraction Fails**
- Ensure PDF is not encrypted
- Try using a different PDF library (pdfplumber)

4. **Audio Generation Too Slow**
- Use caching (enabled by default)
- Process in smaller chunks
- Consider using a faster TTS engine

5. **Session State Lost**
- Normal Streamlit behavior on page refresh
- Implement persistent storage (database)

### Maintenance

#### Regular Tasks

1. **Clean Cache** (Monthly)
```python
import time
from pathlib import Path

def clean_old_cache(cache_dir, days=30):
    """Remove cache files older than X days"""
    now = time.time()
    cutoff = now - (days * 86400)

    for file in Path(cache_dir).glob('*.mp3'):
        if file.stat().st_mtime < cutoff:
            file.unlink()
            logger.info(f"Deleted old cache: {file}")
```

2. **Backup Library** (Weekly)
```python
import shutil
from datetime import datetime

def backup_library():
    """Create backup of library.json"""
    source = 'data/library.json'
    backup = f'backups/library_{datetime.now():%Y%m%d_%H%M%S}.json'
    shutil.copy(source, backup)
    logger.info(f"Library backed up to {backup}")
```

3. **Monitor Disk Space**
```python
import shutil

def check_disk_space(path='/'):
    """Check available disk space"""
    total, used, free = shutil.disk_usage(path)
    percent_free = (free / total) * 100

    if percent_free < 10:
        logger.warning(f"Low disk space: {percent_free:.1f}% free")
```

### Support and Resources

- **Documentation**: README.md
- **Configuration**: config.py
- **Issues**: Check logs in logs/ directory
- **Community**: Streamlit forums

---

**Good luck with your deployment!**
