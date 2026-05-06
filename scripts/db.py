#!/usr/bin/env python3
"""
Hermes Memory SQLite Database — unified storage for facts, relations, embeddings.

Replaces: 9 JSONL files + NetworkX graph.json (single-file, ACID, WAL-optimized).

Opus-reviewed schema v1 — mandatory PRAGMAs, partial unique index, FTS5, fact_relations.

Usage:
    from db import get_db
    db = get_db("~/.hermes/memory/hermes_memory.db")
    db.upsert_fact(key="...", category="preference", embedding=[...])
    results = db.search_similar(embedding=[...], top_k=5)
"""

import sqlite3
import struct
import os
import time
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

# ============================================================
# PRAGMAs — mandatory (Opus-approved, must apply on every connection)
# ============================================================
INIT_PRAGMAS = [
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA mmap_size=268435456",      # 256 MB
    "PRAGMA cache_size=-65536",        # 64 MB (negative = KiB)
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
]

# ============================================================
# Schema — Opus-reviewed final
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_hash   TEXT NOT NULL,
    full_key    TEXT,                          -- normalized canonical key
    category    TEXT NOT NULL DEFAULT 'general',
    importance  INTEGER DEFAULT 1,
    valid_from  REAL,                          -- unixepoch('subsec')
    valid_to    REAL,                          -- NULL = still valid
    superseded_by INTEGER REFERENCES facts(id),
    session_id  TEXT,
    source      TEXT,                          -- 'summarizer' | 'manual' | 'decision_engine'
    aliases     TEXT,                          -- JSON array of alternative phrasings
    embedding   BLOB,                          -- fp16 BGE-M3 (1024-dim × 2 bytes = 2048 bytes)
    created_at  REAL NOT NULL DEFAULT (unixepoch('subsec')),
    last_seen_at REAL,
    seen_count  INTEGER DEFAULT 1
);

-- Partial unique index: only one live version of each fact_hash
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_hash_live
    ON facts(fact_hash) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);

-- Directed relations between facts (replaces NetworkX edges)
CREATE TABLE IF NOT EXISTS fact_relations (
    from_id     INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    to_id       INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                 -- 'belongs_to' | 'from_session' | 'similar' | 'supersedes'
    weight      REAL,
    PRIMARY KEY (from_id, to_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_relations_to ON fact_relations(to_id, kind);

-- Track which sessions have been summarized (prevents re-summarization)
CREATE TABLE IF NOT EXISTS summarized_sessions (
    session_id      TEXT PRIMARY KEY,
    summarized_at   REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Generic key-value store for processing state (tracker, last_run, etc.)
CREATE TABLE IF NOT EXISTS processing_state (
    key         TEXT PRIMARY KEY,
    value       TEXT,                          -- JSON
    updated_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Audit log for debugging
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    details     TEXT,
    created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Full-text search (bilingual: Arabic + English)
-- tokenize='unicode61 remove_diacritics 2' handles Arabic diacritics natively
-- prefix='2' enables wildcard searches like 'مشر*'
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    full_key,
    content,                                    -- will store category + aliases for richer matching
    tokenize='unicode61 remove_diacritics 2',
    prefix='2'
);
"""

# ============================================================
# Embedding helpers — fp16 pack/unpack (BGE-M3 1024-dim)
# ============================================================
EMBEDDING_DIM = 1024

def pack_embedding(embedding: List[float]) -> bytes:
    """Pack float32→float16 embedding into BLOB (2048 bytes for 1024-dim)."""
    import numpy as np
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.astype(np.float16).tobytes()

def unpack_embedding(blob: bytes) -> List[float]:
    """Unpack fp16 BLOB→float32 list."""
    import numpy as np
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    return arr.tolist()


# ============================================================
# Connection pool (thread-local singleton)
# ============================================================
_connections: Dict[str, sqlite3.Connection] = {}
_conn_lock = threading.Lock()

class MemoryDB:
    """Hermes Memory SQLite database with full CRUD + vector search."""

    def __init__(self, db_path: str = "~/.hermes/memory/hermes_memory.db"):
        self.db_path = os.path.expanduser(db_path)
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local connection with all PRAGMAs applied."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.execute("PRAGMA cache_size=-65536")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def init(self):
        """Initialize schema (idempotent). Call once at startup."""
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        self.log_event("schema_init", "Schema initialized")
        return self

    def log_event(self, event_type: str, details: str = ""):
        """Write an audit event."""
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO events (event_type, details) VALUES (?, ?)",
                (event_type, details)
            )
            conn.commit()
        except Exception:
            pass  # Never crash on logging failure

    # ============================================================
    # Fact CRUD
    # ============================================================

    def upsert_fact(
        self,
        key: str,
        category: str = "general",
        embedding: Optional[List[float]] = None,
        session_id: str = "",
        source: str = "unknown",
        importance: int = 1,
        aliases: Optional[List[str]] = None,
    ) -> int:
        """
        Insert or update a fact atomically. Returns fact id.

        Strategy:
        1. Compute fact_hash from normalized key (caller must normalize first)
        2. Find existing live fact with same hash → UPDATE + FTS5 refresh
        3. Otherwise → INSERT new + FTS5 insert

        P4: Wrapped in explicit BEGIN/COMMIT/ROLLBACK — UPDATE/INSERT
        and FTS5 operations are atomic. WAL mode ensures readers see
        only committed state.

        Uses unixepoch('subsec') for sub-millisecond precision.
        """
        conn = self._get_conn()
        now = time.time()
        fact_hash = hashlib.md5(key.encode()).hexdigest()
        emb_blob = pack_embedding(embedding) if embedding else None
        aliases_json = json.dumps(aliases or [], ensure_ascii=False)

        conn.execute("BEGIN IMMEDIATE")
        try:
            # Try update existing live fact
            cur = conn.execute(
                """UPDATE facts 
                   SET importance = MAX(importance, ?),
                       seen_count = seen_count + 1,
                       last_seen_at = ?,
                       aliases = CASE 
                           WHEN aliases IS NULL THEN ? 
                           ELSE aliases 
                       END,
                       embedding = COALESCE(?, embedding)
                   WHERE fact_hash = ? AND valid_to IS NULL
                   RETURNING id""",
                (importance, now, aliases_json, emb_blob, fact_hash)
            )
            row = cur.fetchone()

            if row:
                fact_id = row[0]
                # Update FTS5
                conn.execute(
                    "UPDATE facts_fts SET full_key = ?, content = ? WHERE rowid = ?",
                    (key, f"{category} {aliases_json}", fact_id)
                )
            else:
                # Insert new
                cur = conn.execute(
                    """INSERT INTO facts (fact_hash, full_key, category, importance, 
                                          valid_from, session_id, source, aliases, 
                                          embedding, created_at, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       RETURNING id""",
                    (fact_hash, key, category, importance,
                     now, session_id, source, aliases_json,
                     emb_blob, now, now)
                )
                fact_id = cur.fetchone()[0]

                # Insert into FTS5
                conn.execute(
                    "INSERT INTO facts_fts (rowid, full_key, content) VALUES (?, ?, ?)",
                    (fact_id, key, f"{category} {aliases_json}")
                )

            conn.execute("COMMIT")
            return fact_id
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def invalidate_fact(self, fact_id: int, superseded_by: Optional[int] = None):
        """Mark a fact as no longer valid (soft delete, bi-temporal).
        
        P5: Atomic UPDATE + FTS5 DELETE wrapped in transaction.
        """
        conn = self._get_conn()
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (now, superseded_by, fact_id)
            )
            # Remove from FTS5
            conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        self.log_event("fact_invalidated", f"id={fact_id} superseded_by={superseded_by}")

    def get_fact(self, fact_id: int) -> Optional[Dict]:
        """Get a single fact by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM facts WHERE id = ? AND valid_to IS NULL", (fact_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_facts_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """Get all live facts in a category."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE category = ? AND valid_to IS NULL ORDER BY importance DESC LIMIT ?",
            (category, limit)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_facts_by_session(self, session_id: str) -> List[Dict]:
        """Get all facts from a specific session."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE session_id = ? AND valid_to IS NULL", (session_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ============================================================
    # Relations
    # ============================================================

    def add_relation(self, from_id: int, to_id: int, kind: str, weight: float = 1.0):
        """Add a relation between two facts (INSERT OR REPLACE)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO fact_relations (from_id, to_id, kind, weight)
               VALUES (?, ?, ?, ?)""",
            (from_id, to_id, kind, weight)
        )
        conn.commit()

    def remove_relation(self, from_id: int, to_id: int, kind: str):
        """Remove a specific relation."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM fact_relations WHERE from_id = ? AND to_id = ? AND kind = ?",
            (from_id, to_id, kind)
        )
        conn.commit()

    def get_neighbors(self, fact_id: int, kind: Optional[str] = None, max_hops: int = 1) -> List[Dict]:
        """
        Get related facts (1-hop expansion, like old NetworkX neighbors).

        Returns facts with relation metadata.
        """
        conn = self._get_conn()

        if kind:
            rows = conn.execute(
                """SELECT f.*, r.kind, r.weight
                   FROM fact_relations r
                   JOIN facts f ON f.id = r.to_id
                   WHERE r.from_id = ? AND r.kind = ? AND f.valid_to IS NULL
                   ORDER BY r.weight DESC""",
                (fact_id, kind)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT f.*, r.kind, r.weight
                   FROM fact_relations r
                   JOIN facts f ON f.id = r.to_id
                   WHERE r.from_id = ? AND f.valid_to IS NULL
                   ORDER BY r.weight DESC""",
                (fact_id,)
            ).fetchall()

        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d['relation_kind'] = r['kind']
            d['relation_weight'] = r['weight']
            results.append(d)
        return results

    # ============================================================
    # Vector search (cosine similarity — in Python, not SQLite)
    # ============================================================

    def search_similar(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.5,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        Find top-k similar facts via cosine similarity.

        Loads all live embeddings → computes cosine → returns top matches.
        For large datasets (>10K facts), consider FAISS/ScaNN.
        Performance: ~2ms for 1K facts, ~20ms for 10K facts.
        """
        import numpy as np

        conn = self._get_conn()
        query = "SELECT id, full_key, category, importance, embedding FROM facts WHERE valid_to IS NULL AND embedding IS NOT NULL"
        params = ()

        if category:
            query += " AND category = ?"
            params = (category,)

        rows = conn.execute(query, params).fetchall()

        if not rows:
            return []

        # Build matrix
        ids = []
        keys = []
        categories = []
        importances = []
        embs = []

        for r in rows:
            ids.append(r[0])
            keys.append(r[1])
            categories.append(r[2])
            importances.append(r[3])
            embs.append(unpack_embedding(r[4]))

        emb_matrix = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / np.maximum(norms, 1e-10)

        q = np.asarray(embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-10)

        sims = emb_matrix @ q
        top_indices = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim < threshold:
                break
            results.append({
                'id': int(ids[idx]),
                'key': keys[idx],
                'category': categories[idx],
                'importance': importances[idx],
                'similarity': sim,
            })

        return results

    # ============================================================
    # Full-text search
    # ============================================================

    def search_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Full-text search via FTS5 (bilingual: Arabic + English)."""
        conn = self._get_conn()
        # Add prefix wildcard for partial word matching
        fts_query = ' OR '.join(f'"{t}"*' for t in query.split() if len(t) >= 2)

        if not fts_query:
            return []

        try:
            rows = conn.execute(
                """SELECT f.id, f.full_key, f.category, f.importance, f.aliases,
                          snippet(facts_fts, 1, '<b>', '</b>', '...', 40) AS snippet
                   FROM facts_fts ft
                   JOIN facts f ON f.id = ft.rowid
                   WHERE facts_fts MATCH ? AND f.valid_to IS NULL
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit)
            ).fetchall()

            return [{
                'id': r[0],
                'key': r[1],
                'category': r[2],
                'importance': r[3],
                'aliases': json.loads(r[4]) if r[4] else [],
                'snippet': r[5],
            } for r in rows]
        except sqlite3.OperationalError:
            return []

    # ============================================================
    # Session tracking
    # ============================================================

    def is_session_summarized(self, session_id: str) -> bool:
        """Check if a session has already been summarized."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM summarized_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def mark_session_summarized(self, session_id: str):
        """Mark a session as summarized."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO summarized_sessions (session_id) VALUES (?)",
            (session_id,)
        )
        conn.commit()

    # ============================================================
    # Processing state (key-value store)
    # ============================================================

    def get_state(self, key: str) -> Optional[str]:
        """Get a processing state value."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM processing_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str):
        """Set a processing state value."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO processing_state (key, value, updated_at) 
               VALUES (?, ?, unixepoch('subsec'))
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = unixepoch('subsec')""",
            (key, value)
        )
        conn.commit()

    # ============================================================
    # Stats & maintenance
    # ============================================================

    def stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        live = conn.execute("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL").fetchone()[0]
        relations = conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM summarized_sessions").fetchone()[0]
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            'total_facts': total,
            'live_facts': live,
            'archived_facts': total - live,
            'relations': relations,
            'summarized_sessions': sessions,
            'db_size_bytes': db_size,
            'db_size_mb': round(db_size / (1024 * 1024), 2),
        }

    def vacuum(self, full_vacuum: bool = False):
        """Reclaim space and optimize database.
        
        P5: Actually runs VACUUM when full_vacuum=True (weekly cron).
        Default is PRAGMA optimize only (fast, non-blocking, daily).
        """
        conn = self._get_conn()
        conn.execute("PRAGMA optimize")
        conn.execute("PRAGMA analysis_limit=1000")
        conn.execute("PRAGMA optimize")
        
        if full_vacuum:
            conn.execute("VACUUM")
            self.log_event("vacuum", "Full VACUUM executed")
        else:
            self.log_event("vacuum", "PRAGMA optimize (fast)")

    # ============================================================
    # Helpers
    # ============================================================

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert sqlite3.Row to dict, decoding embedding BLOB."""
        d = dict(row)
        # Decode embedding
        if d.get('embedding'):
            d['embedding'] = unpack_embedding(d['embedding'])
        else:
            d['embedding'] = []
        # Parse aliases
        aliases = d.get('aliases', '[]')
        if isinstance(aliases, str):
            try:
                d['aliases'] = json.loads(aliases)
            except (json.JSONDecodeError, TypeError):
                d['aliases'] = []
        return d

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ============================================================
# Singleton helper
# ============================================================
_default_db: Optional[MemoryDB] = None
_db_lock = threading.Lock()

def get_db(db_path: str = "~/.hermes/memory/hermes_memory.db") -> MemoryDB:
    """Get or create the singleton MemoryDB instance."""
    global _default_db
    if _default_db is None:
        with _db_lock:
            if _default_db is None:
                _default_db = MemoryDB(db_path)
                _default_db.init()
    return _default_db


# ============================================================
# CLI — for testing
# ============================================================
if __name__ == "__main__":
    import sys
    db = MemoryDB()
    db.init()
    s = db.stats()
    print(f"Hermes Memory DB: {db.db_path}")
    print(f"  Live facts: {s['live_facts']}")
    print(f"  Relations: {s['relations']}")
    print(f"  DB size: {s['db_size_mb']} MB")
    print("✓ Schema initialized")
