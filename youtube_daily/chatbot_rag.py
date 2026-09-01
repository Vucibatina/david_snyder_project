#!/usr/bin/env python3
"""
chatbot_rag.py — Conversational RAG chatbot over indexed YouTube transcripts.

CLI mode:    python chatbot_rag.py
Server mode: python chatbot_rag.py --server [--port 8080]
"""

import argparse
import sys
from pathlib import Path
from llama_cpp import Llama
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH      = "/Users/vuk/projects/models/llama-2-7b-chat-hf-q4_k_m.gguf"
DB_DIR          = "/Users/vuk/projects/youtube_daily/rag_db"
COLLECTION_NAME = "youtube_transcripts"
EMBED_MODEL     = "all-MiniLM-L6-v2"

TOP_K              = 5      # chunks to retrieve
DISTANCE_THRESHOLD = 0.65   # cosine distance — above this = not relevant enough
MAX_CONTEXT_CHARS  = 2500   # total chars of retrieved context sent to LLM
MAX_TOKENS         = 512    # max tokens in LLM answer
HISTORY_TURNS      = 4      # past (user, assistant) pairs included in prompt
DEFAULT_PORT       = 8080


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_llm():
    print("Loading Llama model (this takes a moment)...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=4, verbose=False)
    print("Model loaded.\n")
    return llm


def load_collection():
    client   = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col      = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    print(f"RAG database loaded — {col.count()} chunks from indexed transcripts.\n")
    return col


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------
def build_retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return question
    last_q, last_a = history[-1]
    return f"{last_q}\n{last_a[:200]}\n{question}"


def retrieve(collection, question: str, history: list[tuple[str, str]] = None) -> list[dict]:
    query   = build_retrieval_query(question, history or [])
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist <= DISTANCE_THRESHOLD:
            chunks.append({"text": doc, "meta": meta, "distance": dist})
    return chunks


def dedupe_sources(chunks: list[dict]) -> list[dict]:
    """Return unique sources as dicts with similarity score and YouTube URL."""
    seen, sources = {}, []
    for c in chunks:
        m   = c["meta"]
        key = m.get("abs_path", "")
        if key in seen:
            continue
        seen[key] = True
        video_id  = Path(key).stem if key else ""
        sources.append({
            "channel":    m.get("channel", ""),
            "title":      m.get("title", ""),
            "date":       m.get("date", ""),
            "video_id":   video_id,
            "url":        f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
            "similarity": round((1 - c["distance"]) * 100, 1),
        })
    return sources


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def build_history_block(history: list[tuple[str, str]]) -> str:
    recent = history[-HISTORY_TURNS:]
    lines  = []
    for user_msg, assistant_msg in recent:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


def build_prompt(context: str, question: str, history: list[tuple[str, str]]) -> str:
    history_section = ""
    if history:
        history_section = f"\nPrevious conversation:\n{build_history_block(history)}\n"
    return (
        f"[INST] You are a helpful assistant. Answer the question using ONLY the context below"
        f" and the conversation history when relevant.\n"
        f"If the context does not contain enough information, say "
        f"\"I don't have enough information on that topic.\"\n"
        f"Be concise and factual.\n"
        f"{history_section}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question} [/INST]"
    )


def ask(
    llm,
    collection,
    question: str,
    history: list[tuple[str, str]] = None,
) -> tuple[str, list[dict]]:
    history = history or []
    chunks  = retrieve(collection, question, history)

    if not chunks:
        return "I don't have information on that topic in the indexed transcripts.", []

    context_parts, total = [], 0
    for c in chunks:
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        snippet = c["text"][:remaining]
        context_parts.append(snippet)
        total += len(snippet)

    context  = "\n---\n".join(context_parts)
    prompt   = build_prompt(context, question, history)
    response = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        stop=["[INST]", "\n\n\n"],
        echo=False,
    )
    answer  = response["choices"][0]["text"].strip()
    sources = dedupe_sources(chunks)
    return answer, sources


# ---------------------------------------------------------------------------
# CLI REPL
# ---------------------------------------------------------------------------
def format_sources_cli(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        score_str = f"{s['similarity']}%"
        parts     = [x for x in [s["channel"], s["title"], s["date"]] if x]
        line      = f"  • {' | '.join(parts)}  [{score_str}]"
        if s["url"]:
            line += f"\n    {s['url']}"
        lines.append(line)
    return "\n".join(lines)


def run_cli(llm, collection):
    print("=" * 60)
    print("  YouTube Transcript RAG Chatbot")
    print("  Type your question and press Enter.")
    print("  Commands: 'quit'/'exit' to stop, '/clear' to reset conversation.")
    print("=" * 60 + "\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "/clear":
            history.clear()
            print("Conversation cleared.\n")
            continue

        answer, sources = ask(llm, collection, question, history)
        history.append((question, answer))

        print(f"\nAssistant: {answer}\n")
        if sources:
            print("Sources:")
            print(format_sources_cli(sources))
        print()


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YouTube RAG Chat</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #0d0d0d; color: #e0e0e0;
  height: 100vh; display: flex; flex-direction: column;
}

/* Header */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 20px; background: #141414; border-bottom: 1px solid #2a2a2a;
  flex-shrink: 0;
}
.header-left { display: flex; align-items: baseline; gap: 10px; }
header h1 { font-size: 1rem; font-weight: 600; color: #fff; }
.subtitle { font-size: 0.75rem; color: #666; }
#clear-btn {
  background: transparent; color: #888; border: 1px solid #444;
  padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 0.8rem;
  transition: all 0.15s;
}
#clear-btn:hover { background: #222; color: #ccc; border-color: #666; }

/* Messages */
#messages {
  flex: 1; overflow-y: auto; padding: 24px 20px;
  display: flex; flex-direction: column; gap: 20px;
}
#messages:empty::before {
  content: "Ask anything about the indexed YouTube transcripts.";
  color: #444; font-size: 0.9rem; margin: auto;
}

.msg { display: flex; flex-direction: column; max-width: 82%; }
.msg.user    { align-self: flex-end;  align-items: flex-end; }
.msg.assistant { align-self: flex-start; align-items: flex-start; }

.bubble {
  padding: 10px 14px; border-radius: 14px;
  line-height: 1.6; font-size: 0.88rem;
}
.user .bubble {
  background: #1d4ed8; color: #fff;
  border-bottom-right-radius: 4px;
}
.assistant .bubble {
  background: #1a1a1a; border: 1px solid #2d2d2d; color: #ddd;
  border-bottom-left-radius: 4px; white-space: pre-wrap;
}

/* Source cards */
.sources { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; width: 100%; }

.source-card {
  background: #111; border: 1px solid #252525; border-radius: 10px;
  padding: 8px 12px; font-size: 0.78rem;
  display: flex; align-items: flex-start; gap: 10px;
}
.score-badge {
  flex-shrink: 0; padding: 3px 8px; border-radius: 5px;
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.02em;
  margin-top: 1px;
}
.score-high { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.score-mid  { background: #1c1107; color: #fb923c; border: 1px solid #431407; }
.score-low  { background: #1c0a0a; color: #f87171; border: 1px solid #450a0a; }

.source-info { flex: 1; min-width: 0; }
.source-channel { color: #666; font-size: 0.7rem; margin-bottom: 2px; }
.source-title { font-weight: 500; }
.source-title a {
  color: #60a5fa; text-decoration: none;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;
}
.source-title a:hover { text-decoration: underline; color: #93c5fd; }
.source-date { color: #555; font-size: 0.68rem; margin-top: 3px; }

/* Thinking indicator */
.thinking {
  color: #555; font-size: 0.85rem; font-style: italic;
  align-self: flex-start; padding: 4px 0;
  display: flex; align-items: center; gap: 6px;
}
.dot-pulse { display: flex; gap: 4px; }
.dot-pulse span {
  width: 5px; height: 5px; background: #555; border-radius: 50%;
  animation: pulse 1.2s ease-in-out infinite;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse { 0%,80%,100% { opacity: 0.2; } 40% { opacity: 1; } }

/* Input row */
#input-row {
  display: flex; gap: 8px; padding: 14px 20px;
  background: #141414; border-top: 1px solid #2a2a2a; flex-shrink: 0;
}
#question {
  flex: 1; background: #1e1e1e; border: 1px solid #3a3a3a;
  color: #e8e8e8; padding: 10px 14px; border-radius: 8px;
  font-size: 0.9rem; outline: none; transition: border-color 0.15s;
}
#question:focus { border-color: #2563eb; }
#question::placeholder { color: #555; }
#send-btn {
  background: #1d4ed8; color: #fff; border: none;
  padding: 10px 22px; border-radius: 8px; cursor: pointer;
  font-size: 0.9rem; font-weight: 500; transition: background 0.15s;
}
#send-btn:hover    { background: #1e40af; }
#send-btn:disabled { background: #1e3358; cursor: not-allowed; color: #6b7280; }
</style>
</head>
<body>
<header>
  <div class="header-left">
    <h1>YouTube RAG Chat</h1>
    <span class="subtitle">Indexed transcript search</span>
  </div>
  <button id="clear-btn" onclick="clearChat()">Clear conversation</button>
</header>
<div id="messages"></div>
<div id="input-row">
  <input id="question" type="text" placeholder="Ask a question about the videos…" autofocus />
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>

<script>
const sessionId = (() => {
  let id = localStorage.getItem('rag_session');
  if (!id) { id = crypto.randomUUID(); localStorage.setItem('rag_session', id); }
  return id;
})();

const messagesEl = document.getElementById('messages');
const questionEl = document.getElementById('question');
const sendBtn    = document.getElementById('send-btn');

questionEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function scoreClass(s) {
  return s >= 60 ? 'score-high' : s >= 42 ? 'score-mid' : 'score-low';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function buildSourceCards(sources) {
  if (!sources || !sources.length) return '';
  const cards = sources.map(s => {
    const badge = `<span class="score-badge ${scoreClass(s.similarity)}">${s.similarity}%</span>`;
    const titleHtml = s.url
      ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title || 'Video')}</a>`
      : esc(s.title || 'Video');
    const dateHtml = s.date ? `<div class="source-date">${esc(s.date)}</div>` : '';
    return `<div class="source-card">
      ${badge}
      <div class="source-info">
        <div class="source-channel">${esc(s.channel || '')}</div>
        <div class="source-title">${titleHtml}</div>
        ${dateHtml}
      </div>
    </div>`;
  }).join('');
  return `<div class="sources">${cards}</div>`;
}

function addMessage(role, text, sources) {
  const div      = document.createElement('div');
  div.className  = `msg ${role}`;
  const srcHtml  = role === 'assistant' ? buildSourceCards(sources) : '';
  div.innerHTML  = `<div class="bubble">${esc(text)}</div>${srcHtml}`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addThinking() {
  const div     = document.createElement('div');
  div.className = 'thinking';
  div.innerHTML = `<div class="dot-pulse"><span></span><span></span><span></span></div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function sendMessage() {
  const q = questionEl.value.trim();
  if (!q || sendBtn.disabled) return;
  questionEl.value  = '';
  sendBtn.disabled  = true;

  addMessage('user', q);
  const thinkingEl = addThinking();

  try {
    const res  = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, session_id: sessionId }),
    });
    const data = await res.json();
    thinkingEl.remove();
    addMessage('assistant', data.answer || data.error || 'No response.', data.sources);
  } catch (err) {
    thinkingEl.remove();
    addMessage('assistant', 'Network error: ' + err.message, []);
  }

  sendBtn.disabled = false;
  questionEl.focus();
}

async function clearChat() {
  await fetch('/api/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  messagesEl.innerHTML = '';
}
</script>
</body>
</html>"""


def run_server(llm, collection, port: int):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        sys.exit("Flask is required for server mode. Run: pip install flask")

    app      = Flask(__name__)
    sessions: dict[str, list] = {}

    @app.route("/")
    def index():
        return _HTML_PAGE

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data      = request.get_json(force=True, silent=True) or {}
        question  = str(data.get("question", "")).strip()
        sid       = str(data.get("session_id", "default"))
        if not question:
            return jsonify({"error": "empty question"}), 400
        history        = sessions.setdefault(sid, [])
        answer, sources = ask(llm, collection, question, history)
        history.append((question, answer))
        return jsonify({"answer": answer, "sources": sources})

    @app.route("/api/clear", methods=["POST"])
    def clear():
        data = request.get_json(force=True, silent=True) or {}
        sid  = str(data.get("session_id", "default"))
        sessions.pop(sid, None)
        return jsonify({"ok": True})

    print(f"Server running at http://127.0.0.1:{port}  (Ctrl-C to stop)\n")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="YouTube transcript RAG chatbot")
    ap.add_argument("--server", action="store_true", help="Run as a web server")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port for server mode (default: {DEFAULT_PORT})")
    args = ap.parse_args()

    llm        = load_llm()
    collection = load_collection()

    if args.server:
        run_server(llm, collection, args.port)
    else:
        run_cli(llm, collection)


if __name__ == "__main__":
    main()
