"""enotropos — Configuration module.

Reads environment variables, corpus path, and defines project-wide constants.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_corpus_path() -> Path:
    path_file = PROJECT_ROOT / "corpus_path.txt"
    if not path_file.exists():
        raise FileNotFoundError(
            f"{path_file} not found. Create it with the path to the PDF corpus."
        )
    corpus_path = Path(path_file.read_text(encoding="utf-8").strip())
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_path}")
    return corpus_path

CORPUS_ROOT = _load_corpus_path()
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHROMA_PATH = DATA_DIR / "chroma_db"

# ---- API Keys ----
OPENCODE_GO_API_KEY = os.getenv("OPENCODE_GO_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# ---- LLM (OpenCode Go / DeepSeek V4 Flash) ----
LLM_BASE_URL = "https://opencode.ai/zen/go/v1"
LLM_MODEL = "deepseek-v4-flash"

# ---- Embeddings (Jina AI) ----
EMBEDDING_BASE_URL = "https://api.jina.ai/v1"
EMBEDDING_MODEL = "jina-embeddings-v3"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 8

# ---- Chunking ----
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100

# ---- Retrieval ----
TOP_K_CHUNKS = 5

# ---- Extraction ----
EXTRACTION_WRITE_IMAGES = False
EXTRACTION_PAGE_CHUNKS = True

# ---- Language detection ----
LANG_DETECT_CHAR_LIMIT = 500

# ---- System prompt ----
SYSTEM_PROMPT = (
    "Ets un expert en denominacions d'origen protegides (DOP) i indicacions "
    "geografiques protegides (IGP) de vins europeus. Respon les preguntes "
    "basant-te exclusivament en els documents oficials proporcionats com a context. "
    "Cita sempre la font (denominacio, document, seccio) de cada dada que "
    "proporcionis. Si el context no conte la informacio necessaria, digues-ho "
    "clarament. Respon en l'idioma en que es formula la pregunta."
)

SUPPORTED_LANGUAGES = {
    "ca": "Catalan",
    "es": "Spanish",
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "de": "German",
}
