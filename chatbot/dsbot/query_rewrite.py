"""Rewrites a follow-up question into a standalone query for retrieval.

Retrieval embeds only the literal text of the current question. A follow-up
like "can you say more about that?" carries no retrievable content on its own
— this resolves pronouns/anaphora against recent history before we embed,
so retrieval fetches chunks about the right topic instead of latching onto
whatever's superficially similar to the bare follow-up text.
"""
from __future__ import annotations

from typing import List, Tuple

import chatbot_config as cfg
from dsbot import llm

_CONDENSE_SYSTEM = (
    "You rewrite a user's latest chat message into a standalone question that "
    "makes sense with no other context, using the conversation history to fill "
    'in anything implied (e.g. "that technique", "tell me more", "it"). '
    "Preserve the original meaning and intent exactly. Output ONLY the rewritten "
    "question, nothing else — no preamble, no quotes, no explanation."
)


def condense(history: List[Tuple[str, str]], question: str) -> str:
    if not history:
        return question

    transcript_lines = []
    for user_text, assistant_text in history[-4:]:
        transcript_lines.append(f"User: {user_text}")
        transcript_lines.append(f"Assistant: {assistant_text}")
    transcript = "\n".join(transcript_lines)

    messages = [
        {"role": "system", "content": _CONDENSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript}\n\n"
                f"Latest message: {question}\n\n"
                "Rewritten standalone question:"
            ),
        },
    ]
    rewritten = llm.generate(messages, max_tokens=cfg.CONDENSE_MAX_TOKENS).text.strip()
    return rewritten or question
