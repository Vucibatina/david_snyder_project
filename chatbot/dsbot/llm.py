"""Runs chat completions against whichever model is configured — a local GGUF
file via llama-cpp-python, or a cloud model via the Anthropic or OpenAI API.

Which model gets loaded is chosen at runtime via configure(model_key) (see
main.py's --model flag). Each model's chatbot_config.AVAILABLE_MODELS entry
has a provider ("local" / "anthropic" / "openai") that selects the backend;
local entries additionally have a chat_style:

  "qwen3" - prompts are rendered manually (ChatML) instead of going through
            llama-cpp-python's create_chat_completion/auto chat-template
            detection, so the assistant turn can be seeded with a pre-closed,
            empty <think></think> block. This is Qwen3's actual mechanism for
            hard-disabling its reasoning mode (the same trick vLLM/SGLang use
            for enable_thinking=False) — unlike the "/no_think" text
            convention, the model can't second-guess text that's already in
            its own context as a closed block.
  "auto"  - plain instruct models (Llama, Mistral, ...) have no reasoning
            phase to suppress, so we just use create_chat_completion and let
            llama-cpp-python auto-detect the model's own chat template.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chatbot_config as cfg

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_client = None
_model_key = cfg.DEFAULT_MODEL


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


def configure(model_key: str) -> None:
    """Select which model to (lazily) load. Must be called before get_client()/
    generate() if you want anything other than cfg.DEFAULT_MODEL.
    """
    global _model_key, _client
    cfg.model_info(model_key)  # raises if unknown
    if _client is not None and model_key != _model_key:
        raise RuntimeError("Cannot switch models after the client has already been loaded.")
    _model_key = model_key


def _load_local(model_path: str):
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=cfg.N_CTX,
        n_gpu_layers=cfg.N_GPU_LAYERS,
        verbose=False,
    )


def _load_anthropic():
    import anthropic

    return anthropic.Anthropic()


def _load_openai():
    import openai

    return openai.OpenAI()


def get_client():
    global _client
    if _client is None:
        info = cfg.model_info(_model_key)
        provider = info["provider"]
        if provider == "local":
            _client = _load_local(str(info["path"]))
        elif provider == "anthropic":
            _client = _load_anthropic()
        elif provider == "openai":
            _client = _load_openai()
    return _client


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _render_qwen3_prompt(messages: List[dict]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    # Pre-close the think block ourselves: hard-disables reasoning mode.
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _split_system(messages: List[dict]) -> Tuple[Optional[str], List[dict]]:
    """Anthropic takes `system` as a separate top-level param, not a message.
    persona.build_messages() always puts the system prompt first.
    """
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def _generate_local(messages: List[dict], max_tokens: int) -> GenResult:
    client = get_client()
    chat_style = cfg.model_info(_model_key)["chat_style"]

    if chat_style == "qwen3":
        response = client.create_completion(
            prompt=_render_qwen3_prompt(messages),
            max_tokens=max_tokens,
            temperature=cfg.GENERATION_TEMPERATURE,
            stop=["<|im_end|>", "<|im_start|>"],
        )
        raw_text = response["choices"][0]["text"]
    else:
        response = client.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=cfg.GENERATION_TEMPERATURE,
        )
        raw_text = response["choices"][0]["message"]["content"]

    usage = response.get("usage", {})
    return GenResult(
        text=_strip_think_tags(raw_text),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )


def _generate_anthropic(messages: List[dict], max_tokens: int, model_id: str) -> GenResult:
    client = get_client()
    system, chat_messages = _split_system(messages)

    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=chat_messages,
        temperature=cfg.GENERATION_TEMPERATURE,
    )
    raw_text = "".join(block.text for block in response.content if block.type == "text")
    return GenResult(
        text=_strip_think_tags(raw_text),
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
    )


def _generate_openai(messages: List[dict], max_tokens: int, model_id: str) -> GenResult:
    client = get_client()

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=messages,
        temperature=cfg.GENERATION_TEMPERATURE,
    )
    raw_text = response.choices[0].message.content
    return GenResult(
        text=_strip_think_tags(raw_text),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )


def generate(messages: List[dict], max_tokens: Optional[int] = None) -> GenResult:
    info = cfg.model_info(_model_key)
    max_tokens = max_tokens or cfg.GENERATION_MAX_TOKENS

    if info["provider"] == "local":
        return _generate_local(messages, max_tokens)
    if info["provider"] == "anthropic":
        return _generate_anthropic(messages, max_tokens, info["model_id"])
    if info["provider"] == "openai":
        return _generate_openai(messages, max_tokens, info["model_id"])
    raise ValueError(f"Unknown provider: {info['provider']}")
