"""Retrieval: embeds the user's question, queries the RAG project's Chroma
collection, and resolves matched chunks to their parent-window context text.

Reuses the sibling RAG project's already-built code directly (no duplicated
retrieval logic) via a sys.path insertion.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chatbot_config as cfg

# Chroma still attempts a posthog telemetry call even with anonymized_telemetry=False
# (already set in the RAG project's client), and the installed posthog version's
# capture() signature no longer matches what this chromadb version expects — so every
# attempt throws and gets logged as an error. Harmless, but noisy; silence it here
# since it's an upstream chromadb/posthog version mismatch, not something either
# project's own settings can fix.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

sys.path.insert(0, str(cfg.RAG_PROJECT_ROOT))

from src import embedding as rag_embedding  # noqa: E402
from src import state as rag_state  # noqa: E402
from src import vectorstore as rag_vectorstore  # noqa: E402


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float


def warmup() -> None:
    """Eagerly load the embedding model so the first question isn't slow."""
    rag_embedding.get_model()


def retrieve(query_text: str) -> Tuple[List[Hit], Dict[str, str], List[str]]:
    """Return (hits, parent_texts, parent_order) for the given question.

    - hits: matched chunks above the similarity floor, sorted best-first
    - parent_texts: parent_id -> full parent-window text, for context expansion
    - parent_order: unique parent_ids in best-similarity-first order, capped to
      chatbot_config.MAX_CONTEXT_PARENTS
    """
    query_embedding = rag_embedding.embed_query(query_text)
    results = rag_vectorstore.query(query_embedding, top_k=cfg.RETRIEVAL_TOP_K)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hits: List[Hit] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metadatas, distances):
        similarity = 1 - dist
        if similarity < cfg.MIN_SIMILARITY:
            continue
        hits.append(Hit(chunk_id=chunk_id, text=doc, metadata=meta, similarity=similarity))

    hits.sort(key=lambda h: h.similarity, reverse=True)

    parent_order: List[str] = []
    seen = set()
    for hit in hits:
        parent_id = hit.metadata.get("parent_id")
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent_order.append(parent_id)
    parent_order = parent_order[: cfg.MAX_CONTEXT_PARENTS]

    with rag_state.connect() as conn:
        parent_texts = rag_state.get_parent_texts(conn, parent_order)

    return hits, parent_texts, parent_order
