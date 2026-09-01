"""Tunable parameters for the David Snyder chatbot."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Path to the sibling RAG project whose code/vector store we reuse ---
RAG_PROJECT_ROOT = Path("/Users/vuk/projects/david_snyder_project/rag")

# --- Model ---
MODELS_DIR = Path("/Users/vuk/projects/models")

# Each entry is a model you can pick with `python main.py --model <key>` (or
# download with `python download_model.py --model <key>`, for local models).
#
# `provider` selects which backend dsbot/llm.py talks to:
#   "local"     - a GGUF file run locally via llama-cpp-python. Needs `repo_id`
#                 + `filename` (for download_model.py) and a `chat_style`:
#                   "qwen3" - manually rendered ChatML with a pre-closed
#                             <think></think> block, to hard-disable Qwen3's
#                             hybrid reasoning mode (see dsbot/llm.py).
#                   "auto"  - use llama-cpp-python's normal
#                             create_chat_completion / auto chat-template
#                             detection. Fine for any non-reasoning instruct
#                             model (Llama, Mistral, etc.).
#   "anthropic" - Claude via the Anthropic API. Needs `model_id` and the
#                 ANTHROPIC_API_KEY env var (a Claude API key from
#                 console.anthropic.com — NOT a claude.ai/Pro login).
#   "openai"    - ChatGPT via the OpenAI API. Needs `model_id` and the
#                 OPENAI_API_KEY env var (an API key from platform.openai.com —
#                 NOT a chatgpt.com/Plus login).
# Cloud entries need internet and incur a small per-message API cost; local
# entries are free and fully offline once downloaded.
AVAILABLE_MODELS = {
    "qwen3-14b": dict(
        provider="local",
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf",
        chat_style="qwen3",
        description="Local, 14B, highest quality, slowest (~50-60s/answer)",
    ),
    "llama3.1-8b": dict(
        provider="local",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        chat_style="auto",
        description="Local, 8B, no reasoning overhead, ~2x faster, some quality trade-off",
    ),
    "claude": dict(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        description="Cloud (Anthropic), fast + cheap (~$0.01/msg), needs ANTHROPIC_API_KEY + internet",
    ),
    "chatgpt": dict(
        provider="openai",
        model_id="gpt-4o-mini",
        description="Cloud (OpenAI), fast + cheap, needs OPENAI_API_KEY + internet",
    ),
}
DEFAULT_MODEL = "qwen3-14b"

# Which env var each cloud provider needs, and where to generate it (distinct
# from the consumer claude.ai/chatgpt.com login, even under the same account).
PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDER_API_KEY_URL = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
}


def model_info(key: str) -> dict:
    if key not in AVAILABLE_MODELS:
        available = ", ".join(AVAILABLE_MODELS)
        raise ValueError(f"Unknown model '{key}'. Available: {available}")
    info = dict(AVAILABLE_MODELS[key])
    if info["provider"] == "local":
        info["path"] = MODELS_DIR / info["filename"]
    return info


N_CTX = 32768  # native context window for these models (see plan notes on what this actually means)
N_GPU_LAYERS = -1  # offload all layers to Metal
GENERATION_MAX_TOKENS = 500
GENERATION_TEMPERATURE = 0.35
CONDENSE_MAX_TOKENS = 80  # the follow-up rewrite only ever needs one short sentence

# --- Diagnostics ---
SHOW_TIMING = True  # print a per-turn timing/token-count line after each answer

# --- Retrieval ---
RETRIEVAL_TOP_K = 6
MIN_SIMILARITY = 0.30  # chunks below this cosine similarity are dropped as noise
MAX_CONTEXT_PARENTS = 6  # cap on unique parent windows included per turn (quality, not capacity)

# --- Persona / grounding ---
TEACHER_NAME = "Dr. David Snyder"
