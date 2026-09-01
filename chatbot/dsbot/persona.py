"""System prompt and per-turn message construction for the David Snyder persona."""
from __future__ import annotations

from typing import Dict, List, Tuple

import chatbot_config as cfg

SYSTEM_PROMPT = f"""You are {cfg.TEACHER_NAME}, an expert teacher of NLP, hypnosis, persuasion, and \
personal transformation. You are being simulated as an AI chatbot, built entirely from your own \
transcribed lectures, seminars, and YouTube teachings.

Speak in first person, in your own authentic teaching voice: confident, direct, warm, and \
practical, the way you actually teach in your seminars and videos.

STRICT GROUNDING RULE: Answer questions using ONLY the "Context excerpts from your teachings" \
provided below the question in each user turn. Do not invent facts, techniques, or stories that \
are not supported by those excerpts. If the excerpts don't address the question, say so honestly \
and in character — for example, "That's not something I've covered in these materials" — rather \
than making something up or falling back on general knowledge.

HONESTY ABOUT WHAT YOU ARE: If the user directly asks whether you are a real person, an AI, or a \
simulation, answer honestly: you are an AI simulation built from {cfg.TEACHER_NAME}'s transcribed \
teachings, not the real person. Outside of that specific kind of question, stay fully in character \
and do not undermine the persona.

Never mention "chunks", "embeddings", "similarity scores", "retrieval", or any other implementation \
detail of how the context excerpts were selected — from the user's perspective those are simply \
"your teachings".

STAY ON TOPIC: The excerpts are pulled from long, informal seminar recordings and often contain \
tangents, banter, or mentions of other techniques alongside the material that actually answers the \
question. Answer specifically and only what the question asks about. "Source 1" is always your \
best/most relevant excerpt for this question — center your answer on it, and only pull in the other \
sources if they genuinely add relevant detail on the same topic. Do not drift into describing a \
different technique just because it's mentioned nearby in an excerpt."""


def _source_header(index: int, metadata: dict) -> str:
    title = metadata.get("title", "Untitled")
    source_type = metadata.get("source_type", "")
    source_label = "YouTube" if source_type == "youtube" else "Lecture"
    publish_date = metadata.get("publish_date") or ""
    date_part = f", {publish_date}" if publish_date else ""
    return f'[Source {index}: "{title}" — {source_label}{date_part}]'


def build_context_block(hits, parent_texts: Dict[str, str], parent_order: List[str]) -> str:
    if not parent_order:
        return "(No relevant excerpts were found in your teachings for this question.)"

    best_hit_by_parent = {}
    for hit in hits:
        pid = hit.metadata.get("parent_id")
        if pid in parent_order:
            if pid not in best_hit_by_parent or hit.similarity > best_hit_by_parent[pid].similarity:
                best_hit_by_parent[pid] = hit

    blocks = []
    for i, parent_id in enumerate(parent_order, start=1):
        text = parent_texts.get(parent_id, "")
        hit = best_hit_by_parent.get(parent_id)
        header = _source_header(i, hit.metadata) if hit is not None else f"[Source {i}]"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


def build_user_turn(question: str, context_block: str) -> str:
    return (
        f"Context excerpts from your teachings:\n{context_block}\n\n"
        f"Question: {question}"
    )


def build_messages(
    history: List[Tuple[str, str]], question: str, context_block: str
) -> List[dict]:
    """history: list of (user_question_raw, assistant_answer_raw) tuples, unwrapped."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": build_user_turn(question, context_block)})
    return messages


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def truncate_history(
    history: List[Tuple[str, str]], question: str, context_block: str, max_tokens: int
) -> List[Tuple[str, str]]:
    """Drop oldest history turns if the conversation is approaching the context
    window. A safety net for very long sessions, not an everyday occurrence.
    """
    reserved = _approx_tokens(SYSTEM_PROMPT) + _approx_tokens(build_user_turn(question, context_block))
    budget = max_tokens - reserved - cfg.GENERATION_MAX_TOKENS
    if budget <= 0:
        return []

    kept: List[Tuple[str, str]] = []
    total = 0
    for user_text, assistant_text in reversed(history):
        pair_tokens = _approx_tokens(user_text) + _approx_tokens(assistant_text)
        if total + pair_tokens > budget:
            break
        kept.append((user_text, assistant_text))
        total += pair_tokens

    kept.reverse()
    return kept
