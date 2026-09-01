#!/usr/bin/env python3
"""
rag_text.py — Index transcript .txt files into ChromaDB for RAG.

Default data dir: /Users/vuk/projects/youtube_daily/data
Each .txt file is expected to have a 3-line header:
    Line 1: channel name
    Line 2: date
    Line 3: video title
    Rest:   transcript text

Usage:
    python rag_text.py                   # index default dir
    python rag_text.py --dir /other/dir  # index a different dir
    python rag_text.py --reset           # wipe everything and re-index from scratch
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/Users/vuk/projects/youtube_daily/data"
DB_DIR           = "/Users/vuk/projects/youtube_daily/rag_db"
MANIFEST_PATH    = "/Users/vuk/projects/youtube_daily/rag_db/processed_files.json"
COLLECTION_NAME  = "youtube_transcripts"
EMBED_MODEL      = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 1000   # characters — stays well under the 256-token model limit
CHUNK_OVERLAP = 150    # character overlap between consecutive chunks
SAVE_EVERY    = 100    # flush manifest to disk after this many newly indexed files


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    p = Path(MANIFEST_PATH)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)


def file_sig(path: Path) -> list:
    """Return [mtime, size] as a stable fingerprint for change detection."""
    s = path.stat()
    return [s.st_mtime, s.st_size]


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------
def chunk_id(abs_path: str, chunk_idx: int) -> str:
    """Stable, short ChromaDB document ID derived from file path + chunk index."""
    h = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    return f"{h}_{chunk_idx}"


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------
def parse_file(path: Path) -> tuple[str, str, str, str]:
    """Return (channel, date, title, transcript)."""
    text  = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split("\n", 3)
    channel    = parts[0].strip() if len(parts) > 0 else ""
    date       = parts[1].strip() if len(parts) > 1 else ""
    title      = parts[2].strip() if len(parts) > 2 else ""
    transcript = parts[3].strip() if len(parts) > 3 else ""
    return channel, date, title, transcript


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def make_chunks(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def remove_file_chunks(collection, abs_path: str):
    """Delete every chunk previously indexed from this file."""
    results = collection.get(where={"abs_path": abs_path})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def upsert_chunks(collection, ids, documents, metadatas, batch=500):
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start : start + batch],
            documents=documents[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )


# ---------------------------------------------------------------------------
# Core indexing
# ---------------------------------------------------------------------------
def index_one(collection, path: Path, manifest: dict) -> bool:
    """
    Index a single file.  Returns True if the file was (re)indexed,
    False if it was skipped (already up-to-date in the manifest).
    """
    key = str(path)
    sig = file_sig(path)

    cached = manifest.get(key)
    if cached and cached["sig"] == sig:
        return False   # nothing changed — skip

    # File is new or has changed — remove stale chunks if any
    if cached:
        remove_file_chunks(collection, key)

    channel, date, title, transcript = parse_file(path)
    chunks = make_chunks(transcript) if transcript else []

    if chunks:
        ids  = [chunk_id(key, i) for i in range(len(chunks))]
        meta = [
            {
                "abs_path":     key,
                "channel":      channel,
                "date":         date,
                "title":        title,
                "chunk_index":  i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]
        upsert_chunks(collection, ids, chunks, meta)

    manifest[key] = {"sig": sig, "chunks": len(chunks)}
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Index transcript .txt files into ChromaDB (incremental, restartable)."
    )
    ap.add_argument(
        "--dir",
        default=DEFAULT_DATA_DIR,
        help=f"Root directory to index recursively (default: {DEFAULT_DATA_DIR})",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing index and manifest, then re-index everything from scratch.",
    )
    args = ap.parse_args()

    data_dir = Path(args.dir).resolve()
    if not data_dir.is_dir():
        sys.exit(f"Error: '{data_dir}' is not a directory.")

    Path(DB_DIR).mkdir(parents=True, exist_ok=True)

    # Set up ChromaDB
    client   = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if args.reset:
        print("Clearing existing index and manifest …")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        Path(MANIFEST_PATH).unlink(missing_ok=True)
        print("Done — starting fresh.\n")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    manifest  = load_manifest()
    all_files = sorted(data_dir.rglob("*.txt"))

    print(f"Data dir      : {data_dir}")
    print(f"Files found   : {len(all_files)}")
    print(f"In manifest   : {len(manifest)}")
    print(f"Chunks in DB  : {collection.count()}")
    print()

    new_count = err_count = 0

    with tqdm(all_files, unit="file", dynamic_ncols=True) as bar:
        for path in bar:
            bar.set_postfix_str(f"{path.parent.name}/{path.name}"[:55], refresh=False)
            try:
                if index_one(collection, path, manifest):
                    new_count += 1
                    if new_count % SAVE_EVERY == 0:
                        save_manifest(manifest)
            except Exception as exc:
                err_count += 1
                tqdm.write(f"ERROR  {path}: {exc}")

    save_manifest(manifest)

    skipped = len(all_files) - new_count - err_count
    print(f"\nFinished.")
    print(f"  Newly indexed  : {new_count}")
    print(f"  Skipped        : {skipped}  (already up-to-date)")
    print(f"  Errors         : {err_count}")
    print(f"  Total chunks   : {collection.count()}")
    print(f"  DB location    : {DB_DIR}")


if __name__ == "__main__":
    main()
