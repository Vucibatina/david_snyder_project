"""Sentence-aware chunking with a parent-context window.

Handles transcripts with normal punctuation as well as raw, unpunctuated ASR
captions by force-splitting any pseudo-sentence that runs too long.

Two passes over the same sentence stream, built independently so neither
duplicates content:
  - parent windows (large, for context expansion at query time; never embedded)
  - child chunks   (small, for embedding/retrieval)
Each child is then linked to whichever parent window contains its midpoint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import config

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
    return _tokenizer


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    token_count: int


@dataclass
class ChildChunk:
    chunk_id: str
    chunk_index: int
    parent_id: str
    text: str
    token_count: int


def _split_pseudo_sentences(text: str) -> List[str]:
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return []

    raw_sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    sentences: List[str] = []
    for sentence in raw_sentences:
        if count_tokens(sentence) <= config.FORCE_SPLIT_SENTENCE_TOKEN_THRESHOLD:
            sentences.append(sentence)
            continue
        # No (or too little) punctuation to rely on -- force-split on word
        # boundaries into fixed-size pseudo-sentences.
        words = sentence.split(" ")
        window = config.FORCE_SPLIT_WORD_WINDOW
        for start in range(0, len(words), window):
            piece = " ".join(words[start : start + window]).strip()
            if piece:
                sentences.append(piece)

    return sentences


def _build_windows(
    token_counts: List[int], target_tokens: int, overlap_tokens: int
) -> List[Tuple[int, int]]:
    """Greedily group sentence indices into (start, end) windows by token budget."""
    n = len(token_counts)
    if n == 0:
        return []

    windows: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        cur_tokens = 0
        j = i
        while j < n and (j == i or cur_tokens + token_counts[j] <= target_tokens):
            cur_tokens += token_counts[j]
            j += 1
        windows.append((i, j))

        if j >= n:
            break

        # Step back from j to build ~overlap_tokens of overlap for the next window.
        back_tokens = 0
        k = j
        while k > i and back_tokens < overlap_tokens:
            k -= 1
            back_tokens += token_counts[k]
        i = k if k > i else j  # guarantee forward progress

    return windows


def chunk_document(text: str, file_id: str) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    sentences = _split_pseudo_sentences(text)
    if not sentences:
        return [], []

    token_counts = [count_tokens(s) for s in sentences]

    parent_windows = _build_windows(
        token_counts, config.PARENT_CHUNK_TOKENS, config.PARENT_CHUNK_OVERLAP_TOKENS
    )
    child_windows = _build_windows(
        token_counts, config.CHILD_CHUNK_TOKENS, config.CHILD_CHUNK_OVERLAP_TOKENS
    )

    parents: List[ParentChunk] = []
    for p_idx, (p_start, p_end) in enumerate(parent_windows):
        parent_text = " ".join(sentences[p_start:p_end])
        parents.append(
            ParentChunk(
                parent_id=f"{file_id}::p{p_idx}",
                text=parent_text,
                token_count=sum(token_counts[p_start:p_end]),
            )
        )

    def parent_for_midpoint(c_start: int, c_end: int) -> int:
        mid = (c_start + c_end - 1) / 2
        for idx, (p_start, p_end) in enumerate(parent_windows):
            if p_start <= mid < p_end:
                return idx
        return min(range(len(parent_windows)), key=lambda idx: abs(parent_windows[idx][0] - c_start))

    children: List[ChildChunk] = []
    for c_idx, (c_start, c_end) in enumerate(child_windows):
        child_text = " ".join(sentences[c_start:c_end])
        parent_idx = parent_for_midpoint(c_start, c_end)
        children.append(
            ChildChunk(
                chunk_id=f"{file_id}::c{c_idx}",
                chunk_index=c_idx,
                parent_id=parents[parent_idx].parent_id,
                text=child_text,
                token_count=sum(token_counts[c_start:c_end]),
            )
        )

    return parents, children
