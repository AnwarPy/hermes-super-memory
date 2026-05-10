#!/usr/bin/env python3
"""
Memory System Repair Script — إصلاح شامل لنظام الذاكرة
Fixes:
1. FTS5 Arabic search (normalize + reindex)
2. Backfill JSONL → DB gap (108 missing facts)
3. Delete phantom empty DB file
4. Enrich existing graph nodes with trust_score from DB
"""

import sqlite3
import json
import os
import sys
import glob
import hashlib
import re
import unicodedata

HERMES_DIR = os.path.expanduser("~/.hermes")
MEMORY_STORE_DB = os.path.join(HERMES_DIR, "memory_store.db")
HERMES_MEMORY_DB = os.path.join(HERMES_DIR, "memory", "hermes_memory.db")
PHANTOM_DB = os.path.join(HERMES_DIR, "memory", "memory_store.db")
FACTS_DIR = os.path.join(HERMES_DIR, "memory", "facts_auto")
GRAPH_PATH = os.path.join(HERMES_DIR, "graphs", "hermes-memory", "graph.json")

VALID_CATEGORIES = ("preference", "fact", "decision", "correction",
                    "project", "technical", "personal", "service", "general")

# ============================================================
# Arabic normalization
# ============================================================

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for FTS5 indexing."""
    if not text:
        return text
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
    # Normalize alef variants
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Normalize yeh variants
    text = text.replace('ى', 'ي')
    # Normalize hamza on yeh
    text = text.replace('ئ', 'ي')
    # Normalize teh marbuta
    text = text.replace('ة', 'ه')
    # Normalize kaf
    text = text.replace('ك', 'ك')
    return text.strip()

# ============================================================
# Fix 1: Delete phantom empty DB
# ============================================================

def fix_phantom_db():
    """Delete the phantom empty DB at ~/.hermes/memory/memory_store.db."""
    if os.path.exists(PHANTOM_DB) and os.path.getsize(PHANTOM_DB) == 0:
        os.remove(PHANTOM_DB)
        print(f"  ✅ Deleted phantom empty DB: {PHANTOM_DB}")
        return True
    elif os.path.exists(PHANTOM_DB):
        print(f"  ⚠️  Phantom DB exists but has data ({os.path.getsize(PHANTOM_DB)} bytes) — skipping")
        return False
    print(f"  ✅ No phantom DB found")
    return True

# ============================================================
# Fix 2: FTS5 Arabic normalization + reindex
# ============================================================

def fix_fts5_arabic():
    """Rebuild FTS5 with normalized Arabic content column."""
    print("\n=== Fix 2: FTS5 Arabic Normalization ===")

    db = sqlite3.connect(MEMORY_STORE_DB)

    # Check if content_normalized column exists
    cols = [c[1] for c in db.execute("PRAGMA table_info(facts)").fetchall()]

    if "content_normalized" not in cols:
        print("  Adding content_normalized column...")
        db.execute("ALTER TABLE facts ADD COLUMN content_normalized TEXT DEFAULT ''")

        # Populate normalized content for all facts
        print("  Normalizing Arabic text...")
        rows = db.execute("SELECT fact_id, content FROM facts").fetchall()
        for fid, content in rows:
            normalized = normalize_arabic_text(content)
            db.execute("UPDATE facts SET content_normalized = ? WHERE fact_id = ?",
                      (normalized, fid))
        db.commit()
        print(f"  Normalized {len(rows)} facts")
    else:
        print("  content_normalized column already exists")

    # Rebuild FTS5 to include normalized content
    # First drop the old FTS5 virtual table
    db.execute("DROP TABLE IF EXISTS facts_fts")

    # Recreate with normalized content
    db.execute("""
        CREATE VIRTUAL TABLE facts_fts USING fts5(
            content_normalized,
            tags,
            content='facts',
            content_rowid='fact_id'
        )
    """)

    # Recreate triggers
    db.execute("DROP TRIGGER IF EXISTS facts_ai")
    db.execute("DROP TRIGGER IF EXISTS facts_ad")
    db.execute("DROP TRIGGER IF EXISTS facts_au")

    db.execute("""
        CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, content_normalized, tags)
                VALUES (new.fact_id, new.content_normalized, new.tags);
        END
    """)

    db.execute("""
        CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content_normalized, tags)
                VALUES ('delete', old.fact_id, old.content_normalized, old.tags);
        END
    """)

    db.execute("""
        CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content_normalized, tags)
                VALUES ('delete', old.fact_id, old.content_normalized, old.tags);
            INSERT INTO facts_fts(rowid, content_normalized, tags)
                VALUES (new.fact_id, new.content_normalized, new.tags);
        END
    """)

    # Rebuild the FTS5 index from existing data
    print("  Rebuilding FTS5 index...")
    db.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
    db.commit()

    # Verify
    fts_count = db.execute("SELECT COUNT(*) FROM facts_fts_docsize").fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"  FTS5 docs: {fts_count}, Facts: {total}")

    if fts_count == total:
        print("  ✅ FTS5 sync OK")
    else:
        print(f"  ❌ FTS5 sync MISMATCH: {fts_count} vs {total}")

    # Test Arabic search
    print("\n  Testing Arabic FTS5 search:")
    test_terms = ['ملخص', 'مستخدم', 'قاعدة', 'أنور', 'تحديث', 'الاعدادات', 'telegram', 'multica']
    for term in test_terms:
        normalized_term = normalize_arabic_text(term)
        r = db.execute("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?",
                      [normalized_term]).fetchone()[0]
        like_r = db.execute("SELECT COUNT(*) FROM facts WHERE content_normalized LIKE ?",
                           [f'%{normalized_term}%']).fetchone()[0]
        status = '✅' if r > 0 else '❌'
        print(f"    {status} \"{term}\" (normalized: \"{normalized_term}\"): FTS5={r}, LIKE={like_r}")

    db.close()
    print("  ✅ FTS5 rebuild complete")

# ============================================================
# Fix 3: Backfill JSONL → DB gap
# ============================================================

def backfill_jsonl_to_db():
    """Insert facts from JSONL that are missing from memory_store.db."""
    print("\n=== Fix 3: JSONL → DB Backfill ===")

    db = sqlite3.connect(MEMORY_STORE_DB)

    # Get existing facts
    existing_contents = set()
    for row in db.execute("SELECT content FROM facts"):
        existing_contents.add(row[0])

    # Read all JSONL files
    jsonl_entries = []
    for f in sorted(glob.glob(os.path.join(FACTS_DIR, "*.jsonl"))):
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'key' in data:
                        content = data['key']
                        category = data.get('category', os.path.splitext(os.path.basename(f))[0])
                        importance = data.get('importance', 1)
                        jsonl_entries.append((content, category, importance))
                except json.JSONDecodeError:
                    pass

    # Find missing entries
    missing = [e for e in jsonl_entries if e[0] not in existing_contents]
    print(f"  JSONL entries: {len(jsonl_entries)}")
    print(f"  Already in DB: {len(existing_contents)}")
    print(f"  Missing: {len(missing)}")

    if not missing:
        print("  ✅ No missing facts to backfill")
        db.close()
        return

    # Insert missing facts
    trust_map = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.85, 5: 1.0}
    inserted = 0
    for content, category, importance in missing:
        if category not in VALID_CATEGORIES:
            category = 'general'
        trust = trust_map.get(importance, 0.5)
        normalized = normalize_arabic_text(content)
        try:
            db.execute("""
                INSERT OR IGNORE INTO facts (content, content_normalized, category, tags, trust_score)
                VALUES (?, ?, ?, ?, ?)
            """, (content, normalized, category, '', trust))
            if db.total_changes > 0:
                inserted += 1
        except sqlite3.IntegrityError:
            pass

    db.commit()
    print(f"  ✅ Inserted {inserted} missing facts")

    # Verify new total
    total = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print(f"  New DB total: {total} facts")

    db.close()

# ============================================================
# Fix 4: Enrich graph nodes with trust_score from DB
# ============================================================

def enrich_graph_nodes():
    """Update existing graph nodes with trust_score from memory_store.db."""
    print("\n=== Fix 4: Graph Node Enrichment ===")

    if not os.path.exists(MEMORY_STORE_DB):
        print("  ⚠️  memory_store.db not found — skipping")
        return

    if not os.path.exists(GRAPH_PATH):
        print("  ⚠️  graph.json not found — skipping")
        return

    db = sqlite3.connect(MEMORY_STORE_DB)

    # Build lookup: content → trust_score
    trust_lookup = {}
    for row in db.execute("SELECT content, trust_score, category FROM facts"):
        trust_lookup[row[0]] = {'trust_score': row[1], 'category': row[2]}
    db.close()

    print(f"  DB lookup: {len(trust_lookup)} facts")

    # Load graph
    import networkx as nx
    graph = nx.readwrite.json_graph.node_link_graph(
        json.load(open(GRAPH_PATH)),
        multigraph=False
    )

    enriched = 0
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "fact":
            continue

        # Check if node already has good trust_score
        current_trust = data.get("trust_score")
        content = data.get("content", "")

        if content and content in trust_lookup:
            db_trust = trust_lookup[content]["trust_score"]
            if current_trust is None or current_trust != db_trust:
                data["trust_score"] = db_trust
                # Update importance to match
                importance = round(db_trust * 5)
                data["importance"] = max(data.get("importance", 1), importance)
                enriched += 1

    print(f"  Nodes enriched with trust_score: {enriched}")

    # Save updated graph
    graph_data = nx.readwrite.json_graph.node_link_data(graph)
    with open(GRAPH_PATH, 'w') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Graph saved: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Verify
    nodes_with_trust = sum(1 for _, d in graph.nodes(data=True) if d.get("trust_score") is not None)
    nodes_with_importance = sum(1 for _, d in graph.nodes(data=True) if d.get("importance") is not None)
    print(f"  Nodes with trust_score: {nodes_with_trust}/{graph.number_of_nodes()}")
    print(f"  Nodes with importance: {nodes_with_importance}/{graph.number_of_nodes()}")

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Hermes Memory System Repair")
    print("=" * 60)

    fix_phantom_db()
    fix_fts5_arabic()
    backfill_jsonl_to_db()
    enrich_graph_nodes()

    print("\n" + "=" * 60)
    print("✅ All repairs complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
