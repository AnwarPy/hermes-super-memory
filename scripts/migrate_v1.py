#!/usr/bin/env python3
"""
Hermes Memory Migration v1 — JSONL + NetworkX graph → SQLite.

Migrates:
  - graph.json (678 fact nodes → facts table with fp16 embeddings)
  - graph.json (8058 edges → fact_relations table)
  - .graph_tracker.json → processing_state table
  - .summarizer_tracker.json → summarized_sessions table
  - 9 JSONL files → facts table (deduplicated against graph facts)

Usage:
  python3 migrate_v1.py --dry-run      # Validate without writing
  python3 migrate_v1.py                # Execute migration
  python3 migrate_v1.py --jsonl-only   # Only migrate JSONL files
  python3 migrate_v1.py --graph-only   # Only migrate graph.json

Safety:
  - Dry-run shows exactly what will happen
  - Creates backup of old DB before writing
  - Idempotent — safe to re-run (uses INSERT OR IGNORE)
"""

import json
import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add scripts dir to path for db import
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

from db import MemoryDB, pack_embedding, unpack_embedding
from quality_gates import normalize_arabic_text, is_junk_fact

# ============================================================
# Paths
# ============================================================
GRAPHS_DIR = os.path.expanduser("~/.hermes/graphs/hermes-memory")
GRAPH_PATH = os.path.join(GRAPHS_DIR, "graph.json")
FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
DB_PATH = os.path.expanduser("~/.hermes/memory/hermes_memory.db")
TRACKER_FILE = os.path.expanduser("~/.hermes/memory/.graph_tracker.json")
SUMMARIZER_TRACKER = os.path.expanduser("~/.hermes/memory/.summarizer_tracker.json")
BACKUP_DIR = os.path.expanduser("~/.hermes/memory/backups")

# ============================================================
# Migration: graph.json → SQLite
# ============================================================

def migrate_graph(db: MemoryDB, dry_run: bool = False) -> dict:
    """Migrate graph.json nodes + edges to SQLite."""
    
    if not os.path.exists(GRAPH_PATH):
        print("  ⚠ No graph.json found — skipping graph migration")
        return {'nodes': 0, 'edges': 0, 'categories': 0, 'sessions': 0}

    print(f"\n  📊 Loading graph.json...")
    data = json.load(open(GRAPH_PATH))
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])  # NetworkX uses 'edges', not 'links'
    
    print(f"  Nodes: {len(nodes)} ({sum(1 for n in nodes if n.get('type')=='fact')} facts)")
    print(f"  Edges: {len(edges)}")

    stats = {'nodes': 0, 'edges': 0, 'categories': 0, 'sessions': 0, 'skipped_junk': 0}
    
    # Phase 1: Create all fact nodes first
    id_map = {}  # old_node_id → new SQLite row id
    category_nodes = {}
    session_nodes = {}

    for node in nodes:
        ntype = node.get('type', '')
        node_id = node.get('id', '')
        
        if ntype == 'fact':
            content = node.get('content', '').strip()
            if not content:
                continue
            
            # Skip junk facts during migration
            if is_junk_fact(content):
                stats['skipped_junk'] += 1
                if dry_run:
                    print(f"    [JUNK skip] {content[:60]}")
                continue
            
            category = node.get('category', 'general')
            session_id = node.get('session_id', '')
            importance = node.get('importance', 1)
            aliases = node.get('aliases', [])
            emb = node.get('embedding', [])
            
            # Normalize key before hashing
            key_normalized = normalize_arabic_text(content)
            
            if dry_run:
                stats['nodes'] += 1
                id_map[node_id] = node_id  # For edge counting in dry-run
                continue
            
            try:
                fact_id = db.upsert_fact(
                    key=content,
                    category=category,
                    embedding=emb if len(emb) == 1024 else None,
                    session_id=session_id,
                    source='migration',
                    importance=importance,
                    aliases=aliases,
                )
                id_map[node_id] = fact_id
                stats['nodes'] += 1
            except Exception as e:
                print(f"    ⚠ Failed to insert {node_id}: {e}")
        
        elif ntype == 'category':
            category_nodes[node_id] = node
        elif ntype == 'session':
            session_nodes[node_id] = node

    print(f"  Phase 1: {stats['nodes']} facts migrated ({stats['skipped_junk']} junk skipped)")

    # Phase 2: Migrate edges
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        edge_type = edge.get('type', edge.get('kind', 'similar'))
        weight = edge.get('weight', 1.0)

        source_id = id_map.get(source)
        target_id = id_map.get(target)

        # If target is a category/session node, we don't store it in fact_relations
        # (categories are stored in the facts.category column)
        if target in category_nodes or target in session_nodes:
            continue

        if source_id is None or target_id is None:
            continue  # One of the nodes was skipped

        if dry_run:
            stats['edges'] += 1
            continue

        try:
            db.add_relation(source_id, target_id, edge_type, weight)
            stats['edges'] += 1
        except Exception as e:
            print(f"    ⚠ Failed to add edge {source}→{target}: {e}")

    print(f"  Phase 2: {stats['edges']} edges migrated")

    return stats


# ============================================================
# Migration: JSONL → SQLite
# ============================================================

def migrate_jsonl(db: MemoryDB, dry_run: bool = False) -> dict:
    """Migrate 9 JSONL files to SQLite (dedup against existing graph facts)."""

    if not os.path.exists(FACTS_DIR):
        print("  ⚠ No facts_auto directory — skipping JSONL migration")
        return {'facts': 0, 'duplicates': 0, 'files': 0}

    stats = {'facts': 0, 'duplicates': 0, 'files': 0, 'skipped_junk': 0}

    print(f"\n  📂 Scanning JSONL files in {FACTS_DIR}...")
    
    for filename in sorted(os.listdir(FACTS_DIR)):
        if not filename.endswith('.jsonl'):
            continue

        filepath = os.path.join(FACTS_DIR, filename)
        file_category = filename.replace('.jsonl', '')
        stats['files'] += 1

        count = 0
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(fact, dict):
                    continue

                key = (fact.get('key') or '').strip()
                if not key:
                    continue

                if is_junk_fact(key):
                    stats['skipped_junk'] += 1
                    continue

                category = fact.get('category', file_category)
                session_id = fact.get('session_id', '')
                importance = fact.get('importance', 1)

                if dry_run:
                    count += 1
                    continue

                try:
                    fact_id = db.upsert_fact(
                        key=key,
                        category=category,
                        embedding=None,  # No embeddings in JSONL
                        session_id=session_id,
                        source='jsonl_migration',
                        importance=importance,
                    )
                    count += 1
                except Exception as e:
                    print(f"    ⚠ Failed: {key[:50]}... ({e})")

        stats['facts'] += count
        print(f"    {filename}: {count} facts")

    print(f"\n  JSONL total: {stats['facts']} facts ({stats['skipped_junk']} junk skipped)")
    return stats


# ============================================================
# Migration: Trackers → processing_state
# ============================================================

def migrate_trackers(db: MemoryDB, dry_run: bool = False):
    """Migrate .graph_tracker.json and .summarizer_tracker.json to processing_state."""

    # Graph tracker → processing state
    if os.path.exists(TRACKER_FILE):
        print(f"\n  🔄 Migrating graph tracker...")
        with open(TRACKER_FILE) as f:
            tracker = json.load(f)
        
        hashes = tracker.get('indexed_fact_hashes', [])
        print(f"    Indexed hashes: {len(hashes)}")
        
        if not dry_run:
            db.set_state('graph_tracker_hashes', json.dumps(hashes))
            db.set_state('graph_tracker_count', str(len(hashes)))

    # Summarizer tracker → processing_state
    if os.path.exists(SUMMARIZER_TRACKER):
        print(f"\n  🔄 Migrating summarizer tracker...")
        with open(SUMMARIZER_TRACKER) as f:
            tracker = json.load(f)
        
        sessions = tracker.get('summarized_sessions', [])
        print(f"    Summarized sessions: {len(sessions)}")
        
        if not dry_run:
            for sess_id in sessions:
                db.mark_session_summarized(sess_id)
            db.set_state('summarized_sessions_count', str(len(sessions)))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Migrate Hermes Memory from JSON/NetworkX to SQLite'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate without writing')
    parser.add_argument('--jsonl-only', action='store_true',
                        help='Only migrate JSONL files')
    parser.add_argument('--graph-only', action='store_true',
                        help='Only migrate graph.json')
    parser.add_argument('--db', default=DB_PATH,
                        help=f'SQLite database path (default: {DB_PATH})')
    args = parser.parse_args()

    print("=" * 60)
    print("Hermes Memory Migration v1 — JSON → SQLite")
    print("=" * 60)

    if args.dry_run:
        print("\n  🔍 DRY RUN MODE — no data will be written\n")

    # Init DB
    db_path = os.path.expanduser(args.db)
    
    if not args.dry_run:
        # Backup existing DB if any
        if os.path.exists(db_path):
            os.makedirs(BACKUP_DIR, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(BACKUP_DIR, f"hermes_memory.{ts}.db")
            import shutil
            shutil.copy2(db_path, backup_path)
            print(f"  💾 Backed up existing DB to: {backup_path}")
    
    db = MemoryDB(db_path)
    db.init()

    # Run migrations
    graph_stats = {'nodes': 0, 'edges': 0}
    jsonl_stats = {'facts': 0}

    if not args.jsonl_only:
        graph_stats = migrate_graph(db, args.dry_run)
    
    if not args.graph_only:
        jsonl_stats = migrate_jsonl(db, args.dry_run)

    # Migrate trackers
    if not args.jsonl_only and not args.graph_only:
        migrate_trackers(db, args.dry_run)

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)

    total_facts = graph_stats.get('nodes', 0) + jsonl_stats.get('facts', 0)
    total_edges = graph_stats.get('edges', 0)
    
    print(f"  Facts (from graph):  {graph_stats.get('nodes', 0)}")
    print(f"  Facts (from JSONL):  {jsonl_stats.get('facts', 0)}")
    print(f"  Edges (from graph):  {total_edges}")
    print(f"  Total facts:         {total_facts}")
    print(f"  Junk facts skipped:  {graph_stats.get('skipped_junk', 0) + jsonl_stats.get('skipped_junk', 0)}")
    
    if args.dry_run:
        print(f"\n  📋 DRY RUN — no changes made. Run without --dry-run to execute.")
    else:
        # Verify
        s = db.stats()
        print(f"\n  ✅ Migration verified:")
        print(f"     Live facts:     {s['live_facts']}")
        print(f"     Relations:      {s['relations']}")
        print(f"     DB size:        {s['db_size_mb']} MB")
        print(f"  📍 Database: {db_path}")
        
        # Log completion
        db.log_event('migration_complete', 
                     f'v1 done: {total_facts} facts, {total_edges} edges')
        db.vacuum()

    db.close()


if __name__ == "__main__":
    main()
