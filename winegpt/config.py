"""enotropos — Configuration module.

Reads environment variables, corpus path, and defines project-wide constants.
"""
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Countries ----
# Centralized list of supported countries. The knowledge corpus lives under the
# ``Coneixement`` virtual country. Import this instead of redefining the tuple.
DEFAULT_COUNTRY = "Espanya"
SUPPORTED_COUNTRIES = ("Espanya", "Coneixement")

# ---- Knowledge Graph ----
GRAPH_SCHEMA_VERSION = 1


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


_corpus_root: Path | None = None
_corpus_root_lock = threading.Lock()


def get_corpus_root() -> Path:
    """Return the corpus root directory, resolved lazily and cached.

    Resolution is deferred until first use so that importing ``winegpt``
    modules does not fail merely because ``corpus_path.txt`` is absent — this
    keeps test/CI environments that don't have a corpus importable.
    """
    global _corpus_root
    if _corpus_root is None:
        with _corpus_root_lock:
            if _corpus_root is None:
                _corpus_root = _load_corpus_path()
    return _corpus_root


# Backwards-compatible alias. Accessing ``config.CORPUS_ROOT`` resolves lazily
# via module __getattr__; prefer ``get_corpus_root()`` in new code.
def __getattr__(name: str) -> Any:
    if name == "CORPUS_ROOT":
        return get_corpus_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# These are eagerly defined from PROJECT_ROOT and never depend on the corpus.
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHROMA_PATH = DATA_DIR / "chroma_db"

# ---- API Keys ----
OPENCODE_GO_API_KEY = os.getenv("OPENCODE_GO_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# ---- LLM (OpenCode Go / Zen) ----
# Free models (endpoint /zen/v1): nemotron-3-ultra-free, north-mini-code-free,
#   mimo-v2.5-free, deepseek-v4-flash-free, big-pickle
# Paid models (endpoint /zen/go/v1): deepseek-v4-flash, deepseek-v4-pro, glm-5.2, ...
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/go/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

# ---- Table extraction LLM (meta/llama-3.1-8b-instruct via NVIDIA NIM) ----
TABLE_LLM_MODEL = "meta/llama-3.1-8b-instruct"

# ---- Tables DB ----
TABLES_DB_PATH = DATA_DIR / "tables.db"

# ---- Knowledge Graph ----
GRAPH_PATH = DATA_DIR / "graph.pkl"

# ---- Embeddings ----
# Provider: "nvidia" (NVIDIA NIM, default) or "local" (sentence-transformers).
# When switching providers, reset the ChromaDB collections (dimensions differ).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "nvidia")

# NVIDIA NIM provider
EMBEDDING_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBEDDING_DIMENSIONS = 2048
EMBEDDING_BATCH_SIZE = 8
# Seconds to sleep between embedding batches. NVIDIA NIM typically handles
# higher throughput, so the default is 0. Increase if you hit rate limits.
EMBEDDING_BATCH_SLEEP = float(os.getenv("EMBEDDING_BATCH_SLEEP", "0"))

# Local provider (fastembed via ONNX Runtime — no GPU needed, no API call)
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
)
LOCAL_EMBEDDING_DIMENSIONS = 1024  # intfloat/multilingual-e5-large


def get_embedding_dimensions() -> int:
    """Return the embedding dimensions for the active provider."""
    if EMBEDDING_PROVIDER == "local":
        return LOCAL_EMBEDDING_DIMENSIONS
    return EMBEDDING_DIMENSIONS

# ---- Chunking ----
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
# Parent-Child chunking strategy
# Children are small (searched by vector), parents are large (sent to LLM).
CHILD_CHUNK_SIZE_TOKENS = 200
PARENT_CHUNK_SIZE_TOKENS = 1500

# ---- Retrieval ----
TOP_K_CHUNKS = 5

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
