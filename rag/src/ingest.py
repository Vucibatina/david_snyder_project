"""CLI entrypoint: walk the transcript directories, parse, chunk, embed, and
store into Chroma -- skipping files that were already processed and are
unchanged, per config.STATE_DB_PATH.

Usage:
    python -m src.ingest --dir /Users/vuk/projects/data/DavidSnyder
    python -m src.ingest --dir /path/to/data --force   # reprocess everything
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Tuple

from tqdm import tqdm

import config
from src import chunking, embedding, parsing, state, vectorstore


def iter_source_files(data_dir: Path) -> Iterator[Tuple[Path, str]]:
    for subdir_name, source_kind in (
        (config.YOUTUBE_SUBDIR, "youtube"),
        (config.LECTURE_SUBDIR, "lecture"),
    ):
        root = data_dir / subdir_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.txt")):
            rel_parts = path.relative_to(root).parts[:-1]
            if any(part in config.EXCLUDE_DIR_NAMES for part in rel_parts):
                continue
            yield path, source_kind


def process_file(
    path: Path, source_kind: str, data_dir: Path, conn: sqlite3.Connection, force: bool
) -> str:
    file_relpath = str(path.relative_to(data_dir))
    content_hash = state.sha256_of_file(path)
    record = state.get_file_record(conn, file_relpath)

    if not state.needs_processing(
        record,
        content_hash,
        config.EMBEDDING_MODEL_NAME,
        config.CHILD_CHUNK_TOKENS,
        config.CHILD_CHUNK_OVERLAP_TOKENS,
        force=force,
    ):
        return "skipped"

    if record is not None:
        vectorstore.delete_by_file(file_relpath)
        state.delete_file_data(conn, file_relpath)

    try:
        if source_kind == "youtube":
            doc = parsing.parse_youtube_file(path, file_relpath, config.TEACHER_NAME)
        else:
            doc = parsing.parse_lecture_file(path, file_relpath, config.TEACHER_NAME)

        parents, children = chunking.chunk_document(doc.text, file_id=file_relpath)

        if children:
            texts = [c.text for c in children]
            embeddings = embedding.embed_documents(texts)
            ids = [c.chunk_id for c in children]
            metadatas = [
                {
                    "source_type": doc.source_type,
                    "teacher": doc.teacher,
                    "title": doc.title,
                    "publish_date": doc.publish_date or "",
                    "video_id": doc.video_id or "",
                    "file_name": doc.file_name,
                    "file_relpath": doc.file_relpath,
                    "chunk_index": c.chunk_index,
                    "parent_id": c.parent_id,
                    "token_count": c.token_count,
                }
                for c in children
            ]
            vectorstore.add_chunks(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        state.insert_parent_windows(conn, file_relpath, parents)
        state.upsert_file_record(
            conn,
            file_relpath,
            content_hash,
            doc.source_type,
            config.EMBEDDING_MODEL_NAME,
            config.CHILD_CHUNK_TOKENS,
            config.CHILD_CHUNK_OVERLAP_TOKENS,
            len(children),
            status="success",
        )
        return "processed"
    except Exception as exc:  # keep going on a single bad file
        state.upsert_file_record(
            conn,
            file_relpath,
            content_hash,
            source_kind,
            config.EMBEDDING_MODEL_NAME,
            config.CHILD_CHUNK_TOKENS,
            config.CHILD_CHUNK_OVERLAP_TOKENS,
            0,
            status=f"error: {exc}"[:500],
        )
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Reprocess every file, ignoring the manifest.")
    args = parser.parse_args()

    files = list(iter_source_files(args.dir))
    if not files:
        print(f"No .txt files found under {args.dir}")
        return

    counts = {"processed": 0, "skipped": 0, "error": 0}
    started = time.time()

    with state.connect() as conn:
        for path, source_kind in tqdm(files, desc="Ingesting", unit="file"):
            result = process_file(path, source_kind, args.dir, conn, args.force)
            counts[result] += 1
            conn.commit()

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.1f}s -- processed: {counts['processed']}, "
        f"skipped (unchanged): {counts['skipped']}, errors: {counts['error']}"
    )
    print(f"Chroma collection '{config.COLLECTION_NAME}' now has {vectorstore.count()} chunks.")


if __name__ == "__main__":
    main()
