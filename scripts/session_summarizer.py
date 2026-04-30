#!/usr/bin/env python3
"""
Hermes Session Summarizer — ملخصات الجلسات التلقائية (100% محلي)

يقرأ آخر الجلسات غير الملخصة من state.db
يولّد ملخص + حقائق مستخرجة باستخدام Ollama محلياً
يحفظ في ~/.hermes/memory/summaries/ و ~/.hermes/memory/facts_auto/
"""

import sqlite3
import json
import os
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# Configuration — 100% local
# ============================================================
DB_PATH = os.path.expanduser("~/.hermes/state.db")
SUMMARIES_DIR = os.path.expanduser("~/.hermes/memory/summaries")
FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
TRACKER_FILE = os.path.expanduser("~/.hermes/memory/.summarizer_tracker.json")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"

BATCH_SIZE = 5
MAX_MESSAGES_PER_SESSION = 100

VALID_CATEGORIES = (
    "preference", "fact", "decision", "correction",
    "project", "technical", "personal", "service", "general"
)

# ============================================================
# Tracker (atomic writes)
# ============================================================

def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE) as f:
                data = json.load(f)
            if isinstance(data.get("summarized_sessions"), list):
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"summarized_sessions": []}

def save_tracker(tracker):
    tmp = TRACKER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tracker, f, indent=2)
    os.replace(tmp, TRACKER_FILE)

# ============================================================
# Database queries
# ============================================================

def get_unsummarized_sessions(limit=BATCH_SIZE):
    tracker = load_tracker()
    summarized = tracker.get("summarized_sessions", [])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if summarized:
        placeholders = ",".join(["?" for _ in summarized])
        query = f"""
            SELECT id, source, started_at, ended_at, message_count, title
            FROM sessions
            WHERE id NOT IN ({placeholders})
            AND message_count >= 2
            ORDER BY started_at DESC
            LIMIT ?
        """
        cursor.execute(query, summarized + [limit])
    else:
        cursor.execute("""
            SELECT id, source, started_at, ended_at, message_count, title
            FROM sessions
            WHERE message_count >= 2
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,))

    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_session_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, content, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        LIMIT ?
    """, (session_id, MAX_MESSAGES_PER_SESSION))

    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages

# ============================================================
# Parse LLM JSON safely
# ============================================================

def parse_json_response(content):
    content = content.strip()
    if content.startswith("```"):
        newline_idx = content.find("\n")
        if newline_idx != -1:
            content = content[newline_idx + 1:]
        last_fence = content.rfind("```")
        if last_fence != -1:
            content = content[:last_fence]
        content = content.strip()
    return json.loads(content)

# ============================================================
# Summary generation using LOCAL Ollama (with retry)
# ============================================================

def generate_summary(session_id, messages):
    import urllib.request

    conversation_lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if len(content) > 500:
            content = content[:500] + "... [truncated]"
        if not content.strip():
            continue
        conversation_lines.append(f"[{role}]: {content}")

    conversation_text = "\n".join(conversation_lines)

    prompt = (
        "You are a session summarizer. Analyze this conversation and produce:\n"
        "1. A 3-5 bullet point summary in Arabic (or English if the session is in English)\n"
        "2. Extract important facts as key-value pairs with categories\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"summary": ["point 1", "point 2"], '
        '"facts": [{"key": "fact", "category": "preference"}], '
        '"language": "ar", "importance": 3}\n\n'
        "Categories: preference, fact, decision, correction, project, technical, personal, service\n\n"
        f"Session conversation:\n{conversation_text}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a JSON-only session summarizer. Return ONLY valid JSON, no markdown fences, no explanation. Be concise."
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1500,
        }
    }

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            content = result.get("message", {}).get("content", "")
            return parse_json_response(content)

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  JSON parse failed (attempt {attempt+1}), retrying...")
                payload["options"]["num_predict"] = 2000
                continue
            print(f"  JSON parse failed after {max_retries+1} attempts: {e}")
            return None
        except Exception as e:
            print(f"  Ollama call failed: {e}")
            return None

# ============================================================
# Save results
# ============================================================

def save_summary(session_id, summary_data):
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(FACTS_DIR, exist_ok=True)

    summary_file = os.path.join(SUMMARIES_DIR, f"{session_id}.json")
    summary_data["session_id"] = session_id
    summary_data["summarized_at"] = datetime.now(timezone.utc).isoformat()

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"  Summary saved: {summary_file}")

    facts = summary_data.get("facts", [])
    saved_count = 0
    for fact in facts:
        category = fact.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"
        fact_file = os.path.join(FACTS_DIR, f"{category}.jsonl")

        fact_entry = {
            "key": fact.get("key", ""),
            "category": category,
            "session_id": session_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "importance": summary_data.get("importance", 1),
            "source": "session_summarizer",
        }

        with open(fact_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(fact_entry, ensure_ascii=False) + "\n")
        saved_count += 1

    print(f"  Facts saved: {saved_count}")
    return saved_count

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"Hermes Session Summarizer (LOCAL — {OLLAMA_MODEL})")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        return

    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = [m["name"] for m in json.loads(resp.read()).get("models", [])]
            if not any(OLLAMA_MODEL in m for m in models):
                print(f"ERROR: {OLLAMA_MODEL} not found. Available: {models}")
                return
    except Exception as e:
        print(f"ERROR: Ollama not running: {e}")
        return

    sessions = get_unsummarized_sessions(BATCH_SIZE)
    if not sessions:
        print("No new sessions to summarize.")
        return

    print(f"\nFound {len(sessions)} unsummarized sessions\n")

    summarized_ids = []
    total_facts = 0

    for session in sessions:
        sid = session["id"]
        title = session.get("title") or "(no title)"
        msg_count = session.get("message_count", 0)

        print(f"\nProcessing: {sid} — {title} ({msg_count} messages)")

        messages = get_session_messages(sid)
        if not messages:
            print("  No messages found, skipping")
            continue

        start = time.time()
        summary = generate_summary(sid, messages)
        elapsed = time.time() - start
        print(f"  LLM took: {elapsed:.1f}s")

        if not summary:
            print("  Failed to generate summary, skipping")
            continue

        if not isinstance(summary.get("summary"), list):
            print("  Invalid summary format (no summary list), skipping")
            continue

        n_facts = save_summary(sid, summary)
        total_facts += n_facts
        summarized_ids.append(sid)

    if summarized_ids:
        tracker = load_tracker()
        tracker["summarized_sessions"] = list(set(
            tracker["summarized_sessions"] + summarized_ids
        ))
        save_tracker(tracker)
        print(f"\n✓ Summarized {len(summarized_ids)} sessions, {total_facts} facts total")
    else:
        print("\nNo sessions were successfully summarized.")

if __name__ == "__main__":
    main()
