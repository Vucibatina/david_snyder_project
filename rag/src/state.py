"""SQLite manifest tracking which files have been ingested, so reruns only
process new or changed files, and parent-window text for context expansion.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import config
from src.chunking import ParentChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    source_type TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    num_chunks INTEGER NOT NULL,
    processed_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parent_windows (
    parent_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parent_windows_file_path
    ON parent_windows(file_path);
"""


@dataclass
class FileRecord:
    file_path: str
    content_hash: str
    source_type: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    processed_at: str
    status: str


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def connect(db_path: Path = config.STATE_DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_file_record(conn: sqlite3.Connection, file_path: str) -> Optional[FileRecord]:
    row = conn.execute("SELECT * FROM files WHERE file_path = ?", (file_path,)).fetchone()
    if row is None:
        return None
    return FileRecord(**dict(row))


def needs_processing(
    record: Optional[FileRecord],
    content_hash: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool = False,
) -> bool:
    if force or record is None:
        return True
    return (
        record.content_hash != content_hash
        or record.embedding_model != embedding_model
        or record.chunk_size != chunk_size
        or record.chunk_overlap != chunk_overlap
        or record.status != "success"
    )


def delete_file_data(conn: sqlite3.Connection, file_path: str) -> None:
    conn.execute("DELETE FROM parent_windows WHERE file_path = ?", (file_path,))
    conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))


def insert_parent_windows(conn: sqlite3.Connection, file_path: str, parents: List[ParentChunk]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO parent_windows (parent_id, file_path, text, token_count) "
        "VALUES (?, ?, ?, ?)",
        [(p.parent_id, file_path, p.text, p.token_count) for p in parents],
    )


def upsert_file_record(
    conn: sqlite3.Connection,
    file_path: str,
    content_hash: str,
    source_type: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    num_chunks: int,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO files (
            file_path, content_hash, source_type, embedding_model,
            chunk_size, chunk_overlap, num_chunks, processed_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            content_hash=excluded.content_hash,
            source_type=excluded.source_type,
            embedding_model=excluded.embedding_model,
            chunk_size=excluded.chunk_size,
            chunk_overlap=excluded.chunk_overlap,
            num_chunks=excluded.num_chunks,
            processed_at=excluded.processed_at,
            status=excluded.status
        """,
        (
            file_path,
            content_hash,
            source_type,
            embedding_model,
            chunk_size,
            chunk_overlap,
            num_chunks,
            datetime.now(timezone.utc).isoformat(),
            status,
        ),
    )


def get_parent_texts(conn: sqlite3.Connection, parent_ids: List[str]) -> dict:
    if not parent_ids:
        return {}
    placeholders = ",".join("?" for _ in parent_ids)
    rows = conn.execute(
        f"SELECT parent_id, text FROM parent_windows WHERE parent_id IN ({placeholders})",
        parent_ids,
    ).fetchall()
    return {row["parent_id"]: row["text"] for row in rows}
