"""Tunable parameters for the ingestion pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Source data ---
DEFAULT_DATA_DIR = Path("/Users/vuk/projects/data/DavidSnyder")
YOUTUBE_SUBDIR = "youtube"
LECTURE_SUBDIR = "videos"
EXCLUDE_DIR_NAMES = {"backup"}
TEACHER_NAME = "Dr. David Snyder"

# --- Storage ---
DB_DIR = PROJECT_ROOT / "db"
CHROMA_DIR = DB_DIR / "chroma"
STATE_DB_PATH = DB_DIR / "ingest_state.sqlite"
COLLECTION_NAME = "david_snyder_transcripts"

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_BATCH_SIZE = 32
QUERY_INSTRUCTION_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking (measured in bge tokenizer tokens) ---
FORCE_SPLIT_SENTENCE_TOKEN_THRESHOLD = 60  # pseudo-sentences longer than this get force-split
FORCE_SPLIT_WORD_WINDOW = 40

PARENT_CHUNK_TOKENS = 1200
PARENT_CHUNK_OVERLAP_TOKENS = 100

CHILD_CHUNK_TOKENS = 400
CHILD_CHUNK_OVERLAP_TOKENS = 80

# --- Vector index (Chroma / hnswlib) ---
HNSW_SPACE = "cosine"
HNSW_CONSTRUCTION_EF = 200
HNSW_SEARCH_EF = 100
HNSW_M = 64
