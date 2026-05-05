#!/usr/bin/env python3
"""
Hermes Memory Graph Updater — SQLite Edition

يحل محل NetworkX بالكامل:
- القراءة من SQLite للـ dedup الدلالي.
- الكتابة مباشرة إلى facts + fact_relations + processing_state.
- دعم MEM0_DECISION (P2) عبر memory_decision.py (يستخدم SQLite أيضاً).
- تنظيف العقد اليتيمة عبر SQL queries.

التبعيات: db.py, quality_gates.py, memory_decision.py (اختياري).
تمت إزالة: networkx, graph.json, graph_tracker.json بالكامل.
"""

import json
import os
import hashlib
import time
import sys
import fcntl
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# P0: Import unified quality gates + db + migrate helpers
_script_dir = str(Path(__file__).parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from quality_gates import normalize_arabic_text, is_junk_fact
from db import MemoryDB, pack_embedding, EMBEDDING_DIM, get_db

# ============================================================
# Configuration
# ============================================================
FACTS_DIR = os.path.expanduser("~/.hermes/memory/facts_auto")
DB_PATH = os.path.expanduser("~/.hermes/memory/hermes_memory.db")
PROJECT_NAME = "hermes-memory"

# عتبة التشابه الدلالي للكشف عن التكرار (cosine similarity)
SEMANTIC_DEDUP_THRESHOLD = 0.88

VALID_CATEGORIES = (
    "preference", "fact", "decision", "correction",
    "project", "technical", "personal", "service", "general"
)

# ============================================================
# BGE-M3 Embedding (from unified plugin)
# ============================================================
_embedding_model = None

def get_embedding_model():
    """Load BGE-M3 singleton from unified plugin."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    # Add unified plugin to path
    plugin_dir = os.path.expanduser("~/.hermes/plugins")
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    from unified.embedding_model import EmbeddingModel
    _embedding_model = EmbeddingModel(
        model_name="BAAI/bge-m3",
        device="cpu",
        use_fp16=False,
    )
    return _embedding_model

def get_embedding(text):
    """Get BGE-M3 embedding (1024-dim)."""
    model = get_embedding_model()
    return model.embed_query(text)

# ============================================================
# Read new facts from JSONL (Dual-Write source of truth)
# ============================================================

def read_new_facts(db: MemoryDB):
    """
    يقرأ الحقائق الجديدة من JSONL.
    يتتبع الحقائق المُفهرسة عبر db.processing_state بدلاً من graph_tracker.json.
    """
    # Load tracked hashes from SQLite (migrated from tracker)
    tracker_json = db.get_state('tracker_hashes') or '[]'
    indexed_hashes = set(json.loads(tracker_json))

    new_facts = []

    if not os.path.exists(FACTS_DIR):
        os.makedirs(FACTS_DIR, exist_ok=True)
        return new_facts, indexed_hashes

    for filename in os.listdir(FACTS_DIR):
        if not filename.endswith(".jsonl"):
            continue

        file_category = filename.replace(".jsonl", "")
        filepath = os.path.join(FACTS_DIR, filename)

        with open(filepath, encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
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

                    if "category" not in fact:
                        fact["category"] = file_category

                    # Deduplicate by key text only (not full JSON with timestamp)
                    key = (fact.get("key") or "").strip()
                    if not key:
                        continue

                    # P0-5: Use unified quality gate (single source)
                    if is_junk_fact(key):
                        continue

                    # P0-1: Normalize key before hashing — fixes silent Arabic duplicates
                    key_normalized = normalize_arabic_text(key)
                    fact_hash = hashlib.md5(key_normalized.encode()).hexdigest()

                    if fact_hash not in indexed_hashes:
                        new_facts.append(fact)
                        # NOTE: Do NOT mark hash as indexed here. The hash is registered
                        # in main() ONLY after add_facts_to_db succeeds. This prevents
                        # silent data loss if embedding or graph insertion fails.
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return new_facts, indexed_hashes


# ============================================================
# Orphan node cleanup — remove fact nodes no longer in JSONL files
# ============================================================

def collect_all_fact_keys() -> set:
    """Collect all fact keys from JSONL files (the source of truth)."""
    all_keys = set()
    if not os.path.exists(FACTS_DIR):
        return all_keys

    for filename in os.listdir(FACTS_DIR):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(FACTS_DIR, filename)
        with open(filepath, encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fact = json.loads(line)
                        if not isinstance(fact, dict):
                            continue
                        key = (fact.get("key") or "").strip()
                        if key:
                            all_keys.add(key)
                    except json.JSONDecodeError:
                        continue
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return all_keys

def remove_orphan_nodes(db: MemoryDB):
    """
    Remove facts from SQLite that no longer exist in JSONL files.
    Returns (orphan_count, orphan_hashes) for tracker cleanup.
    """
    all_keys = collect_all_fact_keys()
    if not all_keys:
        print("  No fact keys in JSONL files — skipping orphan cleanup")
        return 0, set()

    # Get all live facts from DB
    conn = db._get_conn()
    rows = conn.execute(
        "SELECT id, full_key FROM facts WHERE valid_to IS NULL"
    ).fetchall()

    orphan_ids = []
    orphan_hashes = set()

    for row in rows:
        fact_id, content = row[0], row[1]
        # Check aliases too
        aliases_json = db.get_state(f'fact_{fact_id}_aliases') or '[]'
        aliases = json.loads(aliases_json) if aliases_json else []
        
        if content not in all_keys and not any(a in all_keys for a in aliases):
            orphan_ids.append(fact_id)
            orphan_hashes.add(hashlib.md5(content.encode()).hexdigest())
            for alias in aliases:
                orphan_hashes.add(hashlib.md5(alias.encode()).hexdigest())

    # Invalidate orphans (soft delete)
    for fact_id in orphan_ids:
        db.invalidate_fact(fact_id)

    # Clean up orphan relations
    if orphan_ids:
        conn.execute(
            f"DELETE FROM fact_relations WHERE from_id IN ({','.join('?'*len(orphan_ids))}) OR to_id IN ({','.join('?'*len(orphan_ids))})",
            orphan_ids + orphan_ids
        )
        conn.commit()

    print(f"  Orphan cleanup: {len(orphan_ids)} facts invalidated")
    return len(orphan_ids), orphan_hashes


# ============================================================
# Add facts to SQLite (replaces add_facts_to_graph)
# ============================================================

def _node_id(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]

def add_facts_to_db(db: MemoryDB, facts: list):
    """
    إضافة الحقائق إلى SQLite مع dedup دلالي.
    
    Returns: (nodes_added, relations_added, nodes_merged)
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    now_ts = time.time()
    nodes_added = 0
    relations_added = 0
    nodes_merged = 0
    new_fact_ids = []

    # ====================================================
    # Pre-load all existing embeddings for dedup
    # ====================================================
    conn = db._get_conn()
    existing_rows = conn.execute(
        "SELECT id, embedding FROM facts WHERE valid_to IS NULL AND embedding IS NOT NULL"
    ).fetchall()

    existing_ids = []
    existing_matrix = None

    if existing_rows:
        from db import unpack_embedding
        embs = [unpack_embedding(r[1]) for r in existing_rows]
        existing_ids = [r[0] for r in existing_rows]
        raw_matrix = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        existing_matrix = raw_matrix / np.maximum(norms, 1e-10)

    # ====================================================
    # Process each fact
    # ====================================================
    for fact in facts:
        key = (fact.get("key") or "").strip()
        category = fact.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"
        session_id = fact.get("session_id", "")
        importance = fact.get("importance", 1)

        if not key:
            continue

        node_id = f"fact_{_node_id(key)}"

        # ====================================================
        # Semantic dedup: ابحث عن fact موجود متشابه دلالياً
        # ====================================================
        duplicate_of = None
        key_for_embedding = normalize_arabic_text(key)
        embedding = get_embedding(key_for_embedding)

        if existing_matrix is not None and len(existing_matrix) > 0:
            q = np.asarray(embedding, dtype=np.float32)
            q = q / max(np.linalg.norm(q), 1e-10)
            sims = existing_matrix @ q
            max_idx = int(np.argmax(sims))
            max_sim = float(sims[max_idx])
            if max_sim >= SEMANTIC_DEDUP_THRESHOLD:
                duplicate_of = existing_ids[max_idx]

        if duplicate_of is not None:
            # دمج: عزّز الموجود بدلاً من إنشاء نود جديد
            conn.execute(
                """UPDATE facts 
                   SET importance = MAX(importance, ?),
                       seen_count = seen_count + 1,
                       last_seen_at = ?
                   WHERE id = ?""",
                (importance, now_ts, duplicate_of)
            )
            # احفظ الصياغة البديلة (مفيد للسياق)
            existing_aliases_json = db.get_state(f'fact_{duplicate_of}_aliases') or '[]'
            existing_aliases = json.loads(existing_aliases_json)
            if key not in existing_aliases:
                existing_aliases.append(key)
                db.set_state(f'fact_{duplicate_of}_aliases', json.dumps(existing_aliases[:10], ensure_ascii=False))
            nodes_merged += 1
        else:
            # P2: Mem0-style decision loop (optional, controlled by env var)
            decision_made = False
            if os.getenv("MEM0_DECISION", "0") == "1":
                try:
                    import importlib.util
                    _md_path = Path(__file__).parent / "memory_decision.py"
                    if _md_path.exists():
                        spec = importlib.util.spec_from_file_location("memory_decision", _md_path)
                        md = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(md)
                        
                        # P2: memory_decision now uses SQLite
                        similar = md.find_similar_facts(key, top_k=3)
                        if similar:
                            decision, target, reason = md.decide(fact, similar)
                            if decision == "noop":
                                nodes_merged += 1
                                decision_made = True
                                print(f"    NOOP: {key[:60]}... ({reason[:80]})")
                            elif decision in ("update", "contradict"):
                                if target:
                                    # Handle update/contradict via db
                                    if decision == "update":
                                        conn.execute(
                                            """UPDATE facts 
                                               SET importance = MAX(importance, ?),
                                                   seen_count = seen_count + 1,
                                                   last_seen_at = ?
                                               WHERE id = ?""",
                                            (importance, now_ts, target)
                                        )
                                        decision_made = True
                                        print(f"    UPDATE: {key[:60]}... ({reason[:80]})")
                                    elif decision == "contradict":
                                        db.invalidate_fact(target, superseded_by=None)  # New ID not yet available
                                        print(f"    CONTRADICT: {key[:60]}... ({reason[:80]})")
                except Exception as e:
                    print(f"    ⚠ Decision engine error: {e}")

            if decision_made:
                continue  # Decision handled — skip normal add

            # إدراج الحقيقة الجديدة
            fact_id = db.upsert_fact(
                key=key,
                category=category,
                embedding=embedding,
                session_id=session_id,
                source='graph_updater',
                importance=importance,
            )
            nodes_added += 1
            new_fact_ids.append(fact_id)

            # أضف للتضمينات اللحظية (لكشف التكرار داخل الدفعة نفسها)
            q_normalized = np.asarray(embedding, dtype=np.float32)
            q_normalized = q_normalized / max(np.linalg.norm(q_normalized), 1e-10)
            if existing_matrix is None:
                existing_matrix = q_normalized.reshape(1, -1)
            else:
                existing_matrix = np.vstack([existing_matrix, q_normalized.reshape(1, -1)])
            existing_ids.append(fact_id)

    # ====================================================
    # روابط التشابه بين النودات الجديدة والموجودة (للاسترجاع لاحقاً)
    # ====================================================
    if new_fact_ids:
        new_embs = []
        for nid in new_fact_ids:
            row = conn.execute("SELECT embedding FROM facts WHERE id = ?", (nid,)).fetchone()
            if row and row[0]:
                from db import unpack_embedding
                new_embs.append((nid, unpack_embedding(row[0])))

        existing_facts = []
        for nid in existing_ids:
            if nid not in new_fact_ids:
                row = conn.execute("SELECT embedding FROM facts WHERE id = ?", (nid,)).fetchone()
                if row and row[0]:
                    from db import unpack_embedding
                    existing_facts.append((nid, unpack_embedding(row[0])))

        if new_embs and existing_facts:
            new_arr = np.array([e for _, e in new_embs])
            old_arr = np.array([e for _, e in existing_facts])

            new_arr = new_arr / np.maximum(np.linalg.norm(new_arr, axis=1, keepdims=True), 1e-10)
            old_arr = old_arr / np.maximum(np.linalg.norm(old_arr, axis=1, keepdims=True), 1e-10)

            sim = new_arr @ old_arr.T

            for i in range(len(new_embs)):
                for j in range(len(existing_facts)):
                    if 0.7 < sim[i][j] < SEMANTIC_DEDUP_THRESHOLD:
                        db.add_relation(new_embs[i][0], existing_facts[j][0], "similar", float(sim[i][j]))
                        relations_added += 1

        # new vs new
        if len(new_embs) > 1:
            arr = np.array([e for _, e in new_embs])
            arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-10)
            sim = arr @ arr.T
            for i in range(len(new_embs)):
                for j in range(i + 1, len(new_embs)):
                    if 0.7 < sim[i][j] < SEMANTIC_DEDUP_THRESHOLD:
                        db.add_relation(new_embs[i][0], new_embs[j][0], "similar", float(sim[i][j]))
                        relations_added += 1

    conn.commit()
    return nodes_added, relations_added, nodes_merged


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"Hermes Memory Graph Updater (SQLite + BGE-M3)")
    print("=" * 60)

    # Init DB
    db = get_db(DB_PATH)
    
    # Load BGE-M3
    print("  Loading BGE-M3 embedding model...")
    start_load = time.time()
    try:
        model = get_embedding_model()
        print(f"  Model loaded in {time.time() - start_load:.1f}s")
    except Exception as e:
        print(f"  ERROR: Failed to load BGE-M3: {e}")
        return

    # Stats before
    s_before = db.stats()
    print(f"  Existing facts: {s_before['live_facts']} live, {s_before['relations']} relations")

    # Read new facts FIRST (before orphan cleanup, so dedup has full DB)
    new_facts, indexed_hashes = read_new_facts(db)

    if not new_facts:
        print("  No new facts to add.")
        # Still run orphan cleanup even if no new facts
        orphan_count, orphan_hashes = remove_orphan_nodes(db)
        if orphan_hashes:
            current = indexed_hashes - orphan_hashes
            db.set_state('tracker_hashes', json.dumps(list(current)))
            db.set_state('tracker_count', str(len(current)))
        if orphan_count:
            print(f"  Orphans cleaned: {orphan_count} facts")
        db.log_event("run_complete", "No new facts")
        return

    print(f"  New facts found: {len(new_facts)}")

    start = time.time()
    nodes_added, edges_added, nodes_merged = add_facts_to_db(db, new_facts)
    elapsed = time.time() - start

    print(f"  Nodes added: {nodes_added}")
    print(f"  Nodes merged (semantic dedup): {nodes_merged}")
    print(f"  Relations added: {edges_added}")
    print(f"  Processing time: {elapsed:.1f}s")

    # Register hashes in tracker ONLY AFTER successful add
    new_hashes = set()
    for fact in new_facts:
        key = (fact.get("key") or "").strip()
        if key:
            new_hashes.add(hashlib.md5(key.encode()).hexdigest())
    indexed_hashes.update(new_hashes)

    # Orphan cleanup AFTER adding — dedup had full DB
    orphan_count, orphan_hashes = remove_orphan_nodes(db)
    if orphan_hashes:
        indexed_hashes -= orphan_hashes

    # Persist tracker state to SQLite
    db.set_state('tracker_hashes', json.dumps(list(indexed_hashes)))
    db.set_state('tracker_count', str(len(indexed_hashes)))
    db.set_state('last_run', datetime.now(timezone.utc).isoformat())

    # Final stats
    s_after = db.stats()
    print(f"\n✓ Graph updated: {s_after['live_facts']} facts, {s_after['relations']} relations")
    print(f"  → Searchable via unified_search() and db.search_similar()")
    db.log_event("run_complete", f"Added {nodes_added} facts, {edges_added} relations")

    # Vacuum periodically (every 10 runs)
    run_count = int(db.get_state('run_count') or '0') + 1
    db.set_state('run_count', str(run_count))
    if run_count % 10 == 0:
        db.vacuum()


if __name__ == "__main__":
    main()
