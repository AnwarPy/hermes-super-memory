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
import fcntl
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ============================================================
# Configuration — 100% local
# ============================================================
DB_PATH = os.path.expanduser("~/.hermes/state.db")
SUMMARIES_DIR = os.path.expanduser("~/.hermes/memory/summaries")
FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
TRACKER_FILE = os.path.expanduser("~/.hermes/memory/.summarizer_tracker.json")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("HERMES_SUMMARIZER_MODEL", "qwen2.5:7b")

BATCH_SIZE = 5
MAX_MESSAGES_PER_SESSION = 100

# Minimum fact key length to avoid useless single-word keys
MIN_FACT_KEY_LENGTH = 15

# P0: Import unified quality gates (single source of truth)
import sys, pathlib
_qg_path = pathlib.Path(__file__).parent
if str(_qg_path) not in sys.path:
    sys.path.insert(0, str(_qg_path))
from quality_gates import (
    normalize_arabic_text as normalize_arabic,
    is_junk_fact,
    is_arabic_heavy,
)

# الجلسة لا تُلخَّص قبل أن تكون "خاملة" (idle) لمدة >= هذه الدقائق منذ آخر رسالة.
# الهدف: تجنّب تلخيص جلسات لا تزال مفتوحة وقد تُضاف لها رسائل جديدة.
# 30 دقيقة كافية لأن:
#   - الـ cron يعمل كل ساعة، فلو المستخدم بدأ منذ 5 دقائق، تُلتقَط في الساعة التالية بعد أن تهدأ.
#   - لو المستخدم في جلسة طويلة نشطة، لن تُلخَّص جزئياً قبل اكتمالها.
# يمكن تجاوز هذا عبر env var: HERMES_SESSION_IDLE_MINUTES=N
SESSION_IDLE_MINUTES = int(os.getenv("HERMES_SESSION_IDLE_MINUTES", "30"))

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
    """Fetch sessions that haven't been summarized yet AND appear quiescent.
    
    Excludes:
    - Sessions already in tracker (summarized_sessions)
    - Sessions with id starting 'cron_' — these are cron job responses
      summarizing themselves, which causes meta-noise loops where the
      summarizer's own output becomes input for the next run.
    - Sessions with fewer than 2 messages (no meaningful content)
    - Sessions with recent activity (last message within SESSION_IDLE_MINUTES)
      → جلسة مفتوحة قد تستقبل رسائل جديدة. تلخيصها الآن يعني:
        (أ) ضياع الرسائل اللاحقة (لن تُعاد معالجة الجلسة)
        (ب) تلخيص ناقص يُنتج حقائق سطحية
    """
    tracker = load_tracker()
    summarized = tracker.get("summarized_sessions", [])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # نحسب عتبة "آخر رسالة قبل X دقيقة" — لو آخر رسالة بعد هذا الوقت، الجلسة "نشطة"
    idle_threshold = (
        datetime.now(timezone.utc) - timedelta(minutes=SESSION_IDLE_MINUTES)
    ).timestamp()

    base_filters = """
        s.message_count >= 2
        AND s.id NOT LIKE 'cron_%'
        AND NOT EXISTS (
            SELECT 1 FROM messages m
            WHERE m.session_id = s.id
            AND m.timestamp > ?
        )
    """

    if summarized:
        # Cap to 500 most recent to avoid SQLITE_LIMIT_VARIABLE_NUMBER (32766/999)
        # 500 is more than enough since we only fetch BATCH_SIZE=5 new sessions
        MAX_IN_LIST = 500
        capped = summarized[-MAX_IN_LIST:]
        placeholders = ",".join(["?" for _ in capped])
        query = f"""
            SELECT s.id, s.source, s.started_at, s.ended_at, s.message_count, s.title
            FROM sessions s
            WHERE s.id NOT IN ({placeholders})
            AND {base_filters}
            ORDER BY s.started_at DESC
            LIMIT ?
        """
        cursor.execute(query, capped + [idle_threshold, limit])
    else:
        query = f"""
            SELECT s.id, s.source, s.started_at, s.ended_at, s.message_count, s.title
            FROM sessions s
            WHERE {base_filters}
            ORDER BY s.started_at DESC
            LIMIT ?
        """
        cursor.execute(query, (idle_threshold, limit))

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
# Parse LLM JSON safely — v4: 3-tier robust parser
# ============================================================

def robust_json_parse(content):
    """Parse LLM JSON with 3 attempts: direct → extract block → fix errors."""
    content = content.strip()
    if not content:
        return None

    # Attempt 1: Direct parse (after removing markdown fences)
    cleaned = content
    if cleaned.startswith("```"):
        idx = cleaned.find("\n")
        if idx != -1:
            cleaned = cleaned[idx + 1:]
        last = cleaned.rfind("```")
        if last != -1:
            cleaned = cleaned[:last]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract first JSON object using brace matching
    brace_count = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start >= 0:
                candidate = content[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Attempt 3: Fix common JSON errors
    fixed = cleaned
    # Replace single quotes with double quotes (naive)
    fixed = re.sub(r"(?<!['\"])\b(\w+)\b\s*:\s*'([^']*)'", r'"\1": "\2"', fixed)
    # Remove trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # Fix unescaped quotes in strings (rough)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed all 3 attempts: {e}")
        return None

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

    # كشف لغة الجلسة تلقائياً (#8)
    arabic_chars = sum(1 for c in conversation_text if '\u0600' <= c <= '\u06FF')
    total_chars = len(conversation_text.strip())
    is_arabic_session = (arabic_chars / max(total_chars, 1)) > 0.3

    lang_instruction = (
        "أجب بالعربية الفصحى فقط. لا تستخدم الإنجليزية إلا للأسماء التقنية."
        if is_arabic_session else
        "Respond in English only."
    )

    # P5: Few-shot examples + structured importance rules for 7B model
    FEW_SHOT_EXAMPLES = (
        "EXAMPLE 1 — GOOD output:\n"
        '{"summary": ['
        '"Fixed elif→if bug in graph_builder.py line 142 that caused O(n²) edge creation",'
        '"Added SIMILAR_EDGE_THRESHOLD=0.82 constant to replace hardcoded 0.70 value",'
        '"Integrated Leiden community detection — detected 70 communities from 738 nodes"'
        '], "facts": ['
        '{"key": "Similar edge threshold is 0.82, defined as SIMILAR_EDGE_THRESHOLD constant", "category": "technical"},'
        '{"key": "Leiden community detection runs after every graph save in save_graph()", "category": "technical"},'
        '{"key": "User prefers concise Arabic responses with technical terminology", "category": "preference"}'
        '], "language": "en", "importance": 4}\n\n'

        "EXAMPLE 2 — BAD output (DO NOT DO THIS):\n"
        '{"summary": ["Sessions summarized: 5", "The system was built"], '
        '"facts": [{"key": "Number of nodes is 941", "category": "technical"}, '
        '{"key": "Total communities: 0", "category": "technical"}], '
        '"language": "en", "importance": 3}\n'
        "Why this is bad: generic bullets, snapshot stats instead of lasting facts.\n\n"

        "IMPORTANCE RULES (use these, do not guess):\n"
        "- 5: Security fixes, API key changes, data loss bugs, breaking changes\n"
        "- 4: User preferences, project decisions, config changes, bug fixes with root cause\n"
        "- 3: Technical implementations, new features, file modifications, deployments\n"
        "- 2: Routine tasks, code reviews, discussions without concrete outcomes\n"
        "- 1: Test messages, acknowledgments, greetings, idle sessions\n"
    )

    prompt = (
        f"You are an expert session summarizer and fact extractor. {lang_instruction}\n\n"

        "TASK 1 — Summary (3-5 bullets):\n"
        "- For EACH bullet answer: WHAT was done? + WHICH files/configs? + WHAT was the result?\n"
        "- BAD: 'Fixed the bug' → GOOD: 'Fixed elif→if bug in graph_builder.py:142 causing O(n²) edges'\n"
        "- BAD: 'System was built' → GOOD: 'Built Flask API on port 8080 with SQLite backend and JWT auth'\n"
        "- Include concrete details: file paths, line numbers, versions, URLs, error messages, solutions.\n\n"

        "TASK 2 — Fact Extraction (key-value pairs with categories):\n"
        "- ATOMICITY: Each fact = ONE atomic fact ONLY. Split multi-fact sentences.\n"
        "- Each fact key must be a COMPLETE SENTENCE with SPECIFIC values.\n"
        "- BAD: 'project_directory' → GOOD: 'Project source code is at /home/anwar/multica-source'\n"
        "- BAD: 'dependencies_installed' → GOOD: 'Installed pnpm 10.28.2, Node.js 22, Docker, PostgreSQL 17'\n"
        "- BAD: 'graph contains 13 nodes' → SKIP (snapshot stats, not a lasting fact)\n"
        "- NO INFERENCE: Extract ONLY explicitly stated info. Never invent versions, paths, or dates.\n"
        "- PRESERVE LITERALS: Never translate or rephrase paths, commands, package names, versions, branch names.\n"
        "- BILINGUAL: Extract facts in the SAME language they were discussed. Do NOT translate.\n"
        "- Exclude: transient stats (node counts, file sizes, timestamps), generic phrases ('help was provided').\n"
        "- Include: file paths, URLs, API endpoints, versions, config values, user preferences, decisions made, errors fixed.\n\n"

        f"{FEW_SHOT_EXAMPLES}"

        "Respond ONLY with valid JSON in this exact format:\n"
        '{"summary": ["specific point 1", "specific point 2"], '
        '"facts": [{"key": "complete sentence with specific value", "category": "technical"}], '
        '"language": "ar", "importance": 3}\n\n'
        "Categories: preference, fact, decision, correction, project, technical, personal, service, general\n\n"
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
            return robust_json_parse(content)

        except urllib.error.URLError as e:
            # Explicit timeout handling
            if hasattr(e, 'reason') and 'timed out' in str(e.reason).lower():
                print(f"  Ollama timeout (attempt {attempt+1}), retrying...")
                if attempt < max_retries:
                    continue
                print(f"  Ollama timeout after {max_retries+1} attempts, skipping session")
                return None
            print(f"  Ollama network error: {e}")
            return None
        except TimeoutError:
            print(f"  Ollama timeout (attempt {attempt+1})")
            if attempt < max_retries:
                continue
            return None
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
# P5: Post-processing importance scorer (overrides model guess)
# ============================================================

# Keywords that signal high-importance sessions
_IMPORTANCE_BOOST_KEYWORDS = {
    'security': 2, 'api_key': 2, 'token': 2, 'secret': 2, 'password': 2,
    'preference': 1, 'prefer': 1, 'decision': 1, 'decided': 1,
    'config': 1, 'configuration': 1, 'settings': 1,
    'bug': 1, 'fix': 1, 'fixed': 1, 'error': 1, 'crash': 1,
    'migration': 1, 'backup': 1, 'restore': 1, 'data loss': 2,
    'breaking': 2, 'breaking change': 2,
    'deploy': 1, 'deployment': 1, 'production': 1,
    'remove': 1, 'delete': 1, 'cleanup': 1,
    'refactor': 1, 'rewrite': 1, 'redesign': 1,
    'threshold': 1, 'community': 1, 'graph': 1,
    'authentication': 1, 'auth': 1,
    'permission': 1, 'access': 1, 'role': 1,
    # Arabic keywords
    'تفضيل': 1, 'قرار': 1, 'خطأ': 1, 'إصلاح': 1, 'أمان': 2,
    'مفتاح': 1, 'نقل': 1, 'نسخة احتياطية': 1, 'إعداد': 1,
}

def compute_importance(session_id, messages, facts, model_importance):
    """Post-process importance score based on session content analysis.
    
    Overrides the model's guess (7B models default to 3 for everything).
    Returns 0-5 integer.
    
    Keyword boost levels:
    - boost=2 (critical): security, API keys, data loss, breaking changes → importance 5
    - boost=1 (important): preferences, config, bugs, decisions → importance 3-4
    - boost=0 (neutral): use model guess
    """
    # Analyze session text
    all_text = ""
    for msg in messages:
        content = msg.get("content", "") or ""
        all_text += " " + content.lower()
    
    # Find the highest boost keyword
    max_boost = 0
    for keyword, boost in _IMPORTANCE_BOOST_KEYWORDS.items():
        if keyword in all_text:
            max_boost = max(max_boost, boost)
    
    # Map boost levels to importance scores
    if max_boost >= 2:
        score = 5  # Critical: security, API keys, breaking changes, data loss
    elif max_boost == 1:
        score = 4  # Important: preferences, config, bugs, decisions
    else:
        score = model_importance  # Neutral: use model's guess
    
    # Penalty for sessions with only greetings/acknowledgments
    if len(all_text.strip().split()) < 50:
        if ('ok' in all_text and 'thanks' in all_text) or \
           ('done' in all_text and 'thank' in all_text):
            score = min(score, 1)
    
    # Ensure at least 2 if we extracted facts
    if facts and score < 2:
        score = 2
    
    return max(0, min(5, score))


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
    if not isinstance(facts, list):
        facts = []
    saved_count = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        # P0-5: Use unified quality gate (replaces duplicate SKIP_PHRASES + length checks)
        key = (fact.get("key") or "").strip()
        if not key or len(key) < MIN_FACT_KEY_LENGTH:
            continue

        # P0-1: Use unified quality gate (Arabic-aware + single skip phrases)
        if is_junk_fact(key):
            continue

        category = fact.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"
        fact_file = os.path.join(FACTS_DIR, f"{category}.jsonl")

        fact_entry = {
            "key": key,
            "category": category,
            "session_id": session_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "importance": summary_data.get("importance", 1),
            "source": "session_summarizer",
        }

        # File locking to prevent race conditions with graph_updater
        with open(fact_file, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(fact_entry, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
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

    # Sanity check: verify messages.timestamp is stored as REAL (Unix epoch)
    # If stored as TEXT (ISO strings), the idle_threshold comparison will silently fail
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT typeof(timestamp) FROM messages WHERE timestamp IS NOT NULL LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] not in ("real", "integer"):
            print(f"ERROR: messages.timestamp is stored as '{row[0]}' (expected REAL/integer). "
                  f"The idle session filter will not work correctly. "
                  f"Run: UPDATE messages SET timestamp = strftime('%s', timestamp) * 1.0;")
            conn.close()
            return
    except sqlite3.OperationalError as e:
        print(f"WARNING: Could not verify timestamp schema: {e}")
    conn.close()

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

    total_facts = 0
    summarized_count = 0

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

        if not isinstance(summary, dict):
            print("  Invalid summary format (not a dict), skipping")
            continue

        if not isinstance(summary.get("summary"), list):
            print("  Invalid summary format (no summary list), skipping")
            continue

        n_facts = save_summary(sid, summary)
        total_facts += n_facts
        summarized_count += 1

        # P5: Override importance with content-based scoring
        if summary and isinstance(summary, dict):
            model_importance = summary.get("importance", 3)
            facts = summary.get("facts", [])
            corrected_importance = compute_importance(sid, messages, facts, model_importance)
            if corrected_importance != model_importance:
                summary["importance"] = corrected_importance
                summary["importance_source"] = "post_processing"
                print(f"  Importance: {model_importance} → {corrected_importance} (post-processed)")

        # Persist tracker immediately per-session — crash-safe, no re-summarize
        tracker = load_tracker()
        if sid not in tracker["summarized_sessions"]:
            tracker["summarized_sessions"].append(sid)
            save_tracker(tracker)

    if summarized_count:
        print(f"\n✓ Summarized {summarized_count} sessions, {total_facts} facts total")
    else:
        print("\nNo sessions were successfully summarized.")

if __name__ == "__main__":
    main()
