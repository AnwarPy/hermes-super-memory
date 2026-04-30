#!/usr/bin/env python3
"""
Hermes Fact Extractor — استخراج الحقائق من الملخصات (100% محلي)

يعمل كمستقل: يقرأ الملخصات من summaries/ ويحسن استخراج الحقائق
أو كمكتبة: يستدعى من session_summarizer.py

يستخدم Ollama محلياً — صفر اتصالات خارجية.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# Configuration
# ============================================================
SUMMARIES_DIR = os.path.expanduser("~/.hermes/memory/summaries")
FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
EXTRACTOR_TRACKER = os.path.expanduser("~/.hermes/memory/.extractor_tracker.json")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("HERMES_SUMMARIZER_MODEL", "qwen2.5:3b")

VALID_CATEGORIES = (
    "preference", "fact", "decision", "correction",
    "project", "technical", "personal", "service", "general"
)

# ============================================================
# Tracker
# ============================================================

def load_tracker():
    if os.path.exists(EXTRACTOR_TRACKER):
        try:
            with open(EXTRACTOR_TRACKER) as f:
                data = json.load(f)
            if isinstance(data.get("processed_summaries"), list):
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"processed_summaries": []}

def save_tracker(tracker):
    tmp = EXTRACTOR_TRACKER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tracker, f, indent=2)
    os.replace(tmp, EXTRACTOR_TRACKER)

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
# Extract facts
# ============================================================

def extract_facts_from_summary(summary_data):
    """Use local LLM to extract deeper facts."""
    import urllib.request

    summary_points = summary_data.get("summary", [])
    existing_facts = summary_data.get("facts", [])
    session_id = summary_data.get("session_id", "")

    summary_text = "\n".join(f"- {s}" for s in summary_points)
    existing_text = "\n".join(f"- {f.get('key', '')}" for f in existing_facts)

    prompt = f"""Given this session summary and existing facts, extract ADDITIONAL deeper facts.

Session summary:
{summary_text}

Existing facts:
{existing_text}

Extract NEW facts NOT in the existing list. Respond ONLY with valid JSON:
{{
  "facts": [
    {{"key": "fact description", "category": "technical"}}
  ]
}}

Categories: preference, fact, decision, correction, project, technical, personal, service"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a JSON-only fact extractor. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500}
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result.get("message", {}).get("content", "")
        parsed = parse_json_response(content)
        facts = parsed.get("facts", [])

        # Validate categories
        for f in facts:
            if f.get("category") not in VALID_CATEGORIES:
                f["category"] = "general"

        return facts

    except Exception as e:
        print(f"  Fact extraction failed: {e}")
        return []

# ============================================================
# Save facts
# ============================================================

def save_facts(facts, session_id):
    """Save facts to category-specific JSONL files with consistent schema."""
    os.makedirs(FACTS_DIR, exist_ok=True)

    saved = 0
    for fact in facts:
        category = fact.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"
        fact_file = os.path.join(FACTS_DIR, f"{category}.jsonl")

        entry = {
            "key": fact.get("key", ""),
            "category": category,
            "session_id": session_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "importance": fact.get("importance", 2),
            "source": "fact_extractor",
        }

        with open(fact_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        saved += 1

    return saved

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"Hermes Fact Extractor (LOCAL — {OLLAMA_MODEL})")
    print("=" * 60)

    if not os.path.exists(SUMMARIES_DIR):
        print("No summaries directory found.")
        return

    tracker = load_tracker()
    processed = set(tracker.get("processed_summaries", []))

    summaries = []
    for f in os.listdir(SUMMARIES_DIR):
        if f.endswith(".json") and f.replace(".json", "") not in processed:
            summaries.append(f)

    if not summaries:
        print("No new summaries to process.")
        return

    print(f"  Found {len(summaries)} unprocessed summaries")

    total_facts = 0
    for fname in sorted(summaries):
        filepath = os.path.join(SUMMARIES_DIR, fname)
        with open(filepath) as f:
            data = json.load(f)

        sid = data.get("session_id", fname.replace(".json", ""))
        print(f"\n  Processing: {sid}")

        start = time.time()
        new_facts = extract_facts_from_summary(data)
        elapsed = time.time() - start
        print(f"    LLM took: {elapsed:.1f}s, found {len(new_facts)} new facts")

        if new_facts:
            saved = save_facts(new_facts, sid)
            total_facts += saved

        processed.add(fname.replace(".json", ""))

    tracker["processed_summaries"] = list(processed)
    save_tracker(tracker)

    print(f"\n✓ Extracted {total_facts} additional facts from {len(summaries)} summaries")

if __name__ == "__main__":
    main()
