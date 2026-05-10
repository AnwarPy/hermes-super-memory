#!/usr/bin/env python3
"""
Migration: hermes_memory.db (old) → memory_store.db (new v3.5)

Mapping:
  old facts.full_key → new facts.content
  old facts.category → new facts.category
  old facts.importance → new facts.trust_score (1→0.3, 2→0.5, 3→0.7, 4→0.85, 5→1.0)
  old facts.aliases → new facts.tags
  old fact_relations → new entities + fact_entities
  old summarized_sessions → tracker JSON
"""

import sqlite3
import json
import re
import os
import sys
from datetime import datetime, timezone

OLD_DB = os.path.expanduser("~/.hermes/memory/hermes_memory.db")
NEW_DB = os.path.expanduser("~/.hermes/memory_store.db")
TRACKER_FILE = os.path.expanduser("~/.hermes/memory/.summarizer_tracker.json")

# Importance → trust_score mapping
IMPORTANCE_MAP = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.85, 5: 1.0}

# Entity extraction patterns
ENTITY_PATTERNS = [
    # @Name / Name (person)
    (r'[@]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 'person'),
    # @BotName
    (r'@([A-Za-z0-9_]+bot)', 'telegram_bot'),
    # Multica, Hermes, etc.
    (r'\b(Multica|Hermes|Telegram|OpenClaw|Claude|Qwen|Gemma|WSL)\b', 'platform'),
    # Port numbers
    (r'(?:port|gateway|api_server)\s+(\d{4,5})', 'port'),
    # File paths
    (r'(/[^\s]{5,})', 'file_path'),
    # Version numbers
    (r'\b(v?\d+\.\d+\.\d+)\b', 'version'),
]


def unix_epoch_to_ts(epoch_val):
    """Convert unix epoch float to ISO timestamp string."""
    try:
        if epoch_val and epoch_val > 1e9:  # valid epoch
            return datetime.fromtimestamp(epoch_val, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OverflowError, OSError):
        pass
    return datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def extract_entities(text):
    """Extract entity names from text using patterns."""
    entities = set()
    for pattern, etype in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group(1) if match.lastindex >= 1 else match.group(0)
            if len(name) >= 2:
                entities.add((name, etype))
    return entities


def migrate():
    old_conn = sqlite3.connect(OLD_DB)
    old_conn.row_factory = sqlite3.Row
    new_conn = sqlite3.connect(NEW_DB)

    stats = {
        'facts_migrated': 0,
        'facts_skipped_dup': 0,
        'entities_created': 0,
        'fact_entity_links': 0,
        'relations_migrated': 0,
        'sessions_migrated': 0,
        'errors': 0,
    }

    # ── Step 1: Migrate facts ──────────────────────────────────────
    print("Step 1: Migrating facts...")
    old_cursor = old_conn.cursor()
    old_cursor.execute("SELECT * FROM facts ORDER BY id")
    old_facts = old_cursor.fetchall()

    all_entities = {}  # (name, type) → entity_id

    for row in old_facts:
        content = row['full_key'] or (row['fact_hash'] if row['fact_hash'] else '')
        if not content:
            continue

        category = row['category'] or 'general'
        importance = row['importance'] if row['importance'] is not None else 1
        trust = IMPORTANCE_MAP.get(importance, 0.5)

        # aliases → tags
        tags = ''
        try:
            aliases = row['aliases']
            if aliases:
                alias_list = json.loads(aliases) if isinstance(aliases, str) else aliases
                tags = ', '.join(str(a) for a in alias_list)
        except (json.JSONDecodeError, TypeError):
            tags = str(aliases) if aliases else ''

        created_at = unix_epoch_to_ts(row['created_at'])
        updated_at = unix_epoch_to_ts(row['last_seen_at'] or row['created_at'])

        try:
            new_conn.execute(
                """INSERT OR IGNORE INTO facts (content, category, tags, trust_score, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (content, category, tags, trust, created_at, updated_at)
            )
            fact_id = new_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            if fact_id > 0:
                stats['facts_migrated'] += 1

                # Extract entities from content
                for name, etype in extract_entities(content):
                    key = (name.lower(), etype)
                    if key not in all_entities:
                        try:
                            new_conn.execute(
                                "INSERT INTO entities (name, entity_type, aliases) VALUES (?, ?, ?)",
                                (name, etype, '')
                            )
                            eid = new_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        except sqlite3.IntegrityError:
                            eid = new_conn.execute(
                                "SELECT entity_id FROM entities WHERE name = ?", (name,)
                            ).fetchone()[0]
                        all_entities[key] = eid

                    new_conn.execute(
                        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (fact_id, all_entities[key])
                    )
                    stats['fact_entity_links'] += 1

        except sqlite3.IntegrityError:
            stats['facts_skipped_dup'] += 1
        except Exception as e:
            stats['errors'] += 1
            print(f"  ERROR migrating fact {row['id']}: {e}")

    stats['entities_created'] = len(all_entities)
    new_conn.commit()
    print(f"  Facts migrated: {stats['facts_migrated']}, skipped: {stats['facts_skipped_dup']}")
    print(f"  Entities created: {stats['entities_created']}")
    print(f"  Fact-entity links: {stats['fact_entity_links']}")

    # ── Step 2: Migrate fact_relations → fact_entities ─────────────
    print("\nStep 2: Migrating relations...")
    old_cursor.execute("SELECT * FROM fact_relations")
    relations = old_cursor.fetchall()

    id_to_content = {}
    for row in old_facts:
        id_to_content[row['id']] = row['full_key'] if row['full_key'] else ''

    for rel in relations:
        kind = rel['kind']
        # Create an entity for the relation type
        entity_name = f"relation:{kind}"
        key = (entity_name, 'relation_type')
        if key not in all_entities:
            try:
                new_conn.execute(
                    "INSERT INTO entities (name, entity_type, aliases) VALUES (?, ?, ?)",
                    (entity_name, 'relation_type', '')
                )
                eid = new_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            except sqlite3.IntegrityError:
                eid = new_conn.execute(
                    "SELECT entity_id FROM entities WHERE name = ?", (entity_name,)
                ).fetchone()[0]
            all_entities[key] = eid

        # Link both from_id and to_id facts to this relation entity
        for old_id in [rel['from_id'], rel['to_id']]:
            content = id_to_content.get(old_id, '')
            if content:
                # Find new fact_id
                result = new_conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                if result:
                    new_fact_id = result[0]
                    new_conn.execute(
                        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (new_fact_id, all_entities[key])
                    )
                    stats['relations_migrated'] += 1

    new_conn.commit()
    print(f"  Relations migrated: {stats['relations_migrated']}")

    # ── Step 3: Migrate summarized_sessions → tracker ──────────────
    print("\nStep 3: Migrating summarized sessions...")
    old_cursor.execute("SELECT session_id FROM summarized_sessions")
    sessions = [row['session_id'] for row in old_cursor.fetchall()]

    if sessions:
        # Load existing tracker
        tracker = {}
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE) as f:
                tracker = json.load(f)

        if 'summarized_sessions' not in tracker:
            tracker['summarized_sessions'] = []

        existing = set(tracker['summarized_sessions'])
        added = 0
        for sid in sessions:
            if sid not in existing:
                tracker['summarized_sessions'].append(sid)
                existing.add(sid)
                added += 1

        with open(TRACKER_FILE, 'w') as f:
            json.dump(tracker, f, ensure_ascii=False, indent=2)

        stats['sessions_migrated'] = added
        print(f"  Sessions added to tracker: {added} (total: {len(sessions)})")

    # ── Step 4: Rebuild FTS5 index ─────────────────────────────────
    print("\nStep 4: Rebuilding FTS5 index...")
    new_conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
    new_conn.commit()
    fts_count = new_conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
    print(f"  FTS5 documents: {fts_count}")

    # ── Final verification ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    new_facts = new_conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    new_entities = new_conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    new_links = new_conn.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]

    print(f"New DB - facts: {new_facts}")
    print(f"New DB - entities: {new_entities}")
    print(f"New DB - fact_entities: {new_links}")

    old_facts_count = old_conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    old_relations = old_conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
    old_sessions = old_conn.execute("SELECT COUNT(*) FROM summarized_sessions").fetchone()[0]

    print(f"\nOld DB - facts: {old_facts_count}")
    print(f"Old DB - relations: {old_relations}")
    print(f"Old DB - sessions: {old_sessions}")

    old_conn.close()
    new_conn.close()

    print("\n✅ Migration complete!")
    return stats


if __name__ == '__main__':
    migrate()
