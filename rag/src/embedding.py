"""Wraps sentence-transformers for computing document/query embeddings."""
from __future__ import annotations

from typing import List

import config

_model = None


def _detect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        device = _detect_device()
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        _model.max_seq_length = config.EMBEDDING_MAX_SEQ_LENGTH
    return _model


def embed_documents(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    prefixed = f"{config.QUERY_INSTRUCTION_PREFIX}{query}"
    return embed_documents([prefixed])[0]
