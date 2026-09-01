"""Thin wrapper around a persistent Chroma collection."""
from __future__ import annotations

from typing import Dict, List, Optional

import config

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={
                "hnsw:space": config.HNSW_SPACE,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
                "hnsw:M": config.HNSW_M,
            },
        )
    return _collection


def delete_by_file(file_relpath: str) -> None:
    collection = get_collection()
    collection.delete(where={"file_relpath": file_relpath})


def add_chunks(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict],
) -> None:
    if not ids:
        return
    collection = get_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def count() -> int:
    return get_collection().count()


def query(embedding: List[float], top_k: int = 5, where: Optional[Dict] = None):
    collection = get_collection()
    return collection.query(query_embeddings=[embedding], n_results=top_k, where=where)
