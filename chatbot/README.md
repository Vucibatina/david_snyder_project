# David Snyder Chatbot

A command-line chatbot that answers questions in the voice of Dr. David Snyder, grounded strictly
in his transcribed lectures and YouTube videos. By default it runs entirely locally — a local LLM
(Qwen3-14B) and a local vector database — no API keys, no internet connection needed once set up,
no per-message cost. Optionally, `--model claude` or `--model chatgpt` route generation through the
Anthropic/OpenAI API instead — much faster, at the cost of an API key, an internet connection, and a
small per-message charge (see "Choosing a model" below).

The persona prompt tells the model which lecture/video each excerpt came from, so answers naturally
cite their sources inline (e.g. "In [lecture], I mentioned..."); there's no separate references list.
Each turn ends with a `====...` separator line so turns are easy to tell apart in the terminal.

## How it works, in short

Your questions are turned into a numeric embedding and matched against ~14,700 pre-indexed excerpts
from ~480 of David Snyder's transcripts (built by the sibling `rag/` project). The best-matching
excerpts are handed to a locally-running LLM along with your question, which answers strictly from
that material, in character, and declines (rather than making something up) if the material doesn't
cover what you asked.

## Prerequisites

- macOS on Apple Silicon (developed/tested on an M3 Max) — Metal GPU offload is used for speed.
  It will still run on Intel Macs or Linux/Windows, just on CPU only, which will be much slower.
- Python 3.10+ (`python3 --version` to check)
- The sibling RAG project at `/Users/vuk/projects/david_snyder_project/rag` must already exist with
  its vector store built (i.e. `rag/db/chroma` and `rag/db/ingest_state.sqlite` populated). This
  chatbot reads that database directly; it does not duplicate or rebuild it.
- ~10GB free disk space for the model file (already downloaded — see below). Not needed for the
  cloud models (`claude` / `chatgpt`) — nothing to download for those.
- ~10-15GB free RAM while running a local model (both the embedding model and the 14B LLM are loaded
  at once). Cloud models use much less — only the (small) embedding model is loaded locally.

## One-time setup

Skip any step that's already done — this repo already has the `.venv` created, dependencies
installed, and the model downloaded, but here's the full setup from scratch for reference:

```bash
cd /Users/vuk/projects/david_snyder_project/chatbot

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (includes llama-cpp-python built with Metal support on macOS,
#    plus the same libraries the rag/ project uses, since this project imports its code)
pip install -r requirements.txt

# 3. Download a model (one-time per model). Defaults to Qwen3-14B (~9GB), saved to
#    /Users/vuk/projects/models/. See "Choosing a model" below for other options.
python download_model.py
```

If step 3 fails partway through (e.g. connection drop), just re-run the same `python
download_model.py` command — it resumes/re-downloads safely and won't create a second copy.

## Running it

Every time you want to talk to the bot:

```bash
cd /Users/vuk/projects/david_snyder_project/chatbot
source .venv/bin/activate
python main.py
```

By default this uses Qwen3-14B. To use a different downloaded model, pass `--model`:

```bash
python main.py --model llama3.1-8b
python main.py --list-models   # see everything available
```

You'll see:

```
Warming up Dr. David Snyder (loading embedding model + LLM, this can take a bit)...

Ready. Talk to Dr. David Snyder. Type 'exit' or 'quit' (or Ctrl-D) to end the session.

You:
```

The warm-up (loading both models into memory) typically takes well under a minute. Then just type
a question and press Enter. Example:

```
You: Tell me about the Pillars of Power technique.

Dr. David Snyder: The Pillars of Power technique is a powerful method used to install and
reinforce specific driver states within yourself, such as playful, curious, and victorious...
[...]

[retrieval 1.12s | generate 24.48s (6290→380 tok) | total 25.60s]

======================================================================

You:
```

(The `[...]` timing line is the per-turn diagnostics described below; set `SHOW_TIMING = False` in
`chatbot_config.py` to hide it. The `====` line always prints, marking the end of the turn.)

The conversation stays continuous — you can ask follow-up questions ("can you say more about
that?") and it will keep track of what you were just discussing. Conversation memory only lasts
for the current run; closing the program clears it.

**To end the session**, type `exit` or `quit`, or press `Ctrl-D`. `Ctrl-C` also works if it ever
gets stuck generating something you don't want to wait for.

**Performance note**: with the default Qwen3-14B (a local model on your GPU, not a cloud API),
expect roughly 50-60 seconds per answer; `llama3.1-8b` is noticeably faster, at some quality cost.
`--model claude` / `--model chatgpt` (cloud) are faster still — typically a few seconds per answer.
Follow-up questions add a couple of seconds on top either way, since the bot does an extra small LLM
call to figure out what a vague follow-up ("that", "the first one") is actually referring to before
it searches the transcripts. Each answer is followed by a `[...]` diagnostics line breaking down
where the time went (retrieval / rewrite / generation, plus token counts) — set `SHOW_TIMING = False`
in `chatbot_config.py` to turn it off.

## Choosing a model

Four models are currently registered in `chatbot_config.AVAILABLE_MODELS`, two local (downloaded
GGUF files, run on your GPU) and two cloud (API calls to Anthropic/OpenAI):

| Key | Where it runs | Notes |
|---|---|---|
| `qwen3-14b` (default) | Local, ~9GB | Highest quality, slowest (~50-60s/answer) |
| `llama3.1-8b` | Local, ~5GB | No reasoning overhead, ~2x faster, some quality trade-off |
| `claude` | Cloud (Anthropic) | Claude Haiku 4.5 — fast (seconds, not tens-of-seconds), ~$0.01/message, needs `ANTHROPIC_API_KEY` + internet |
| `chatgpt` | Cloud (OpenAI) | GPT-4o-mini — fast, needs `OPENAI_API_KEY` + internet |

For a local model, download it once, then pick it at run time:

```bash
python download_model.py --model llama3.1-8b
python main.py --model llama3.1-8b
```

For a cloud model, there's nothing to download — just set the API key (see "Cloud API keys" below)
and pick it at run time:

```bash
python main.py --model claude
python main.py --model chatgpt
```

`python main.py --list-models` / `python download_model.py --list-models` print the current list.
To add another model, add an entry to `AVAILABLE_MODELS` in `chatbot_config.py` — local entries need
`provider: "local"`, a repo ID, a GGUF filename, and `chat_style` (`"auto"` for any plain instruct
model; `"qwen3"` is only needed for Qwen3's own hybrid-reasoning prompt format, see the comment in
`dsbot/llm.py`); cloud entries need `provider: "anthropic"` or `"openai"` and a `model_id`.

### Cloud API keys

`claude` and `chatgpt` need their own **API keys** — these are a separate product from a normal
claude.ai/Claude Pro or chatgpt.com/ChatGPT Plus login, with separate (small, per-token) billing,
even under the same account email:

- **Claude**: generate a key at <https://console.anthropic.com/settings/keys>, then
  `export ANTHROPIC_API_KEY=sk-...`
- **ChatGPT**: generate a key at <https://platform.openai.com/api-keys>, then
  `export OPENAI_API_KEY=sk-...`

Run those `export` lines in the same terminal you'll run `python main.py` from (or add them to your
shell profile to persist across sessions). If the key isn't set, `python main.py --model claude` /
`--model chatgpt` prints exactly this before doing anything else, rather than failing partway
through.

## Project files

| File | What it does |
|---|---|
| `chatbot_config.py` | All the tunable settings: which models are available (`AVAILABLE_MODELS`), context window size, generation temperature, how many source excerpts to retrieve, etc. |
| `download_model.py` | Script to fetch a model's GGUF file from Hugging Face. Takes `--model <key>`. |
| `main.py` | The program you run. Loads everything once, then loops: read your question → look up relevant transcript excerpts → ask the LLM → print the answer → repeat. |
| `dsbot/retrieval.py` | Turns your question into a search and fetches the most relevant transcript excerpts from the vector database (built by the `rag/` project). |
| `dsbot/query_rewrite.py` | Rewrites vague follow-up questions ("what about that?") into standalone questions using the recent conversation, so the search step knows what you actually mean. |
| `dsbot/persona.py` | Builds the actual instructions given to the model: "you are Dr. David Snyder, answer only from the provided excerpts," plus the specific excerpts (with their source lecture/video names) for this question. |
| `dsbot/llm.py` | Runs the selected model — local via `llama-cpp-python` (your Mac's GPU), or cloud via the Anthropic/OpenAI API. |
| `requirements.txt` | The list of Python packages this project needs. |

## Changing settings

Open `chatbot_config.py` to adjust things like:

- `GENERATION_TEMPERATURE` — lower (e.g. 0.1-0.3) makes answers more consistent/literal; higher
  (e.g. 0.6-0.8) makes them more varied/creative. Currently 0.35.
- `GENERATION_MAX_TOKENS` — hard cap on answer length (currently 500). Answers that would run
  longer get cut off, so don't drop this much below the length of a typical full answer.
- `RETRIEVAL_TOP_K` / `MIN_SIMILARITY` — how many transcript excerpts are considered per question,
  and how relevant they must be to count.
- `AVAILABLE_MODELS` / `DEFAULT_MODEL` — see "Choosing a model" above for adding/switching models;
  no need to edit this to just switch between already-downloaded models, use `--model` instead.

No code changes are needed for any of the above — just edit the values and rerun `python main.py`.

## Troubleshooting

- **"No module named ..." errors** — make sure you ran `source .venv/bin/activate` first, in the
  same terminal window you're running `python main.py` from.
- **Model file not found** — `python main.py` prints the exact `download_model.py --model <key>`
  command to run for whichever model you selected (or the default, if you didn't pass `--model`);
  files land in `/Users/vuk/projects/models/`.
- **Very slow / seems frozen** — the first response after starting the program is normal to take a
  bit longer. Check Activity Monitor: the `python3` (actually `main.py`) process should be using
  noticeable GPU (not 0%) — if the fans are spinning and GPU history shows activity, it's working,
  just slow. A totally idle Mac with a stalled process is the real warning sign.
- **Answers seem to reference the wrong topic on a follow-up question** — the source material is
  raw, informal seminar recordings with a lot of tangents mixed into the teaching content, so this
  can occasionally happen. Rephrasing the follow-up as a fully standalone question (naming the
  technique explicitly again) reliably works around it.
- **`Failed to send telemetry event` messages** — harmless; this is Chroma's (the vector database
  library's) own internal analytics call failing silently. Doesn't affect functionality.
- **"Error talking to anthropic/openai: ..." mid-conversation** — only applies to `--model claude` /
  `--model chatgpt`. Usually an expired/invalid API key, no billing/credits set up on that API
  account, a rate limit, or a dropped internet connection. The conversation keeps going — just ask
  your question again once the underlying issue is fixed.
