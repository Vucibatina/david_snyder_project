"""David Snyder chatbot — CLI REPL.

Run with:
    python main.py
    python main.py --model llama3.1-8b

Use --list-models to see what's available.
"""
from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY / OPENAI_API_KEY from a local .env file, if present

import chatbot_config as cfg
from dsbot import llm, persona, query_rewrite, retrieval

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

_CLOUD_ERRORS = tuple(
    exc
    for exc in (
        anthropic.APIError if anthropic else None,
        openai.APIError if openai else None,
    )
    if exc is not None
)

_TURN_SEPARATOR = "=" * 70


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the David Snyder bot.")
    parser.add_argument(
        "--model",
        choices=list(cfg.AVAILABLE_MODELS),
        default=cfg.DEFAULT_MODEL,
        help=f"Which model to use, local or cloud (default: {cfg.DEFAULT_MODEL}). See --list-models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available --model choices and exit.",
    )
    return parser.parse_args()


def _print_model_list() -> None:
    print("Available models:")
    for key, info in cfg.AVAILABLE_MODELS.items():
        marker = " (default)" if key == cfg.DEFAULT_MODEL else ""
        print(f"  {key}{marker} — {info['description']}")


def main() -> None:
    args = _parse_args()
    if args.list_models:
        _print_model_list()
        return

    llm.configure(args.model)
    info = cfg.model_info(args.model)

    if info["provider"] == "local":
        if not info["path"].exists():
            print(f"Model file not found: {info['path']}")
            print(f"Run: python download_model.py --model {args.model}")
            return
    else:
        env_var = cfg.PROVIDER_API_KEY_ENV[info["provider"]]
        if not os.environ.get(env_var):
            print(f"{env_var} is not set, so '{args.model}' can't be used.")
            print(f"Get an API key from {cfg.PROVIDER_API_KEY_URL[info['provider']]}")
            print(f"(this is a separate product from a claude.ai/chatgpt.com login),")
            print(f"then run: export {env_var}=sk-...")
            return

    print(f"Warming up {cfg.TEACHER_NAME} (loading embedding model + LLM: {args.model}, this can take a bit)...")
    retrieval.warmup()
    llm.get_client()
    print(f"\nReady. Talk to {cfg.TEACHER_NAME}. Type 'exit' or 'quit' (or Ctrl-D) to end the session.\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        turn_start = time.perf_counter()

        try:
            # Resolve pronouns/anaphora ("that", "the first one") against history into a
            # standalone question. Used for BOTH retrieval and what the LLM itself sees —
            # otherwise the model can misread an ambiguous "first" against the newly
            # retrieved excerpts instead of against what was actually just discussed.
            rewrite_start = time.perf_counter()
            rewrite_ran = bool(history)
            retrieval_query = query_rewrite.condense(history, question)
            rewrite_elapsed = time.perf_counter() - rewrite_start

            retrieval_start = time.perf_counter()
            hits, parent_texts, parent_order = retrieval.retrieve(retrieval_query)
            retrieval_elapsed = time.perf_counter() - retrieval_start

            context_block = persona.build_context_block(hits, parent_texts, parent_order)

            truncated_history = persona.truncate_history(history, retrieval_query, context_block, cfg.N_CTX)
            messages = persona.build_messages(truncated_history, retrieval_query, context_block)

            generate_start = time.perf_counter()
            result = llm.generate(messages)
            generate_elapsed = time.perf_counter() - generate_start
        except _CLOUD_ERRORS as e:
            print(f"\nError talking to {info['provider']}: {e}\n")
            print(f"{_TURN_SEPARATOR}\n")
            continue

        answer = result.text
        total_elapsed = time.perf_counter() - turn_start

        print(f"\n{cfg.TEACHER_NAME}: {answer}\n")

        if cfg.SHOW_TIMING:
            segments = []
            if rewrite_ran:
                segments.append(f"rewrite {rewrite_elapsed:.2f}s")
            segments.append(f"retrieval {retrieval_elapsed:.2f}s")
            segments.append(
                f"generate {generate_elapsed:.2f}s "
                f"({result.prompt_tokens}→{result.completion_tokens} tok)"
            )
            segments.append(f"total {total_elapsed:.2f}s")
            print(f"[{' | '.join(segments)}]\n")

        print(f"{_TURN_SEPARATOR}\n")

        history.append((question, answer))


if __name__ == "__main__":
    main()
