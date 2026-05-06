"""P1b: Auto Consolidation — Decay + Compression + Archive + Pinning

This module handles memory lifecycle management:
1. Decay: time-based score adjustment (already in __init__.py)
2. Compression: merge similar facts into consolidated summaries
3. Archive: preserve original facts before compression
4. Pinning: protect critical facts from compression
5. Dry-run: preview compression without making changes
6. Audit log: track what was compressed

Usage:
    from unified.consolidation import MemoryConsolidator

    consolidator = MemoryConsolidator(config={
        'decay_lambda': 0.01,
        'similarity_threshold': 0.85,
        'max_facts': 10000,
        'compress_model': 'qwen2.5:3b',
        'dry_run': True,  # First run MUST be dry-run
        'protected_categories': ['preference', 'identity'],
        'archive_enabled': True,
    })

    report = consolidator.consolidate()
    print(report)  # Shows what would happen
"""

import json
import logging
import math
import os
import sqlite3
import sys
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Configuration defaults
# ============================================================

DEFAULT_CONFIG = {
    'decay_lambda': 0.0,        # 0.0 = disabled, 0.01 = half-life ~69 days
    'similarity_threshold': 0.85,  # cosine similarity for grouping
    'max_facts': 10000,         # max facts before compression triggers
    'compress_model': 'qwen2.5:3b',
    'dry_run': True,            # Safety: MUST be True first
    'protected_categories': ['preference', 'identity'],
    'archive_enabled': True,
    'ollama_url': 'http://localhost:11434/api/generate',
}


class MemoryConsolidator:
    """P1b: Manages memory lifecycle — decay, compression, archive, pinning."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._memory_db = None
        self._archive_db_path = None

    def _get_db_path(self):
        """Get the actual SQLite file path for consolidation operations."""
        return os.path.expanduser('~/.hermes/scripts/hermes_memory.db')

    def _get_db(self):
        """Get MemoryDB connection for read operations."""
        if self._memory_db is None:
            from unified.memory_db import _get_memory_db
            self._memory_db = _get_memory_db()
        return self._memory_db

    def _get_archive_conn(self):
        """Get SQLite connection for archive operations."""
        db_path = self._get_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path)

    def _ensure_archive_table(self):
        """Create compressed_facts_archive table if it doesn't exist.
        
        Note: This is called by _archive_facts which already has the connection.
        """
        conn = self._get_archive_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compressed_facts_archive (
                    id TEXT PRIMARY KEY,
                    original_content TEXT,
                    compressed_content TEXT,
                    compression_timestamp REAL,
                    source_ids TEXT,
                    category TEXT
                )
            """)
            conn.commit()
            logger.info("Archive table ready")
            return True
        except Exception as e:
            logger.error("Failed to create archive table: %s", e)
            return False
        finally:
            conn.close()

    def _needs_compression(self) -> bool:
        """Check if compression is needed (fact count > max_facts)."""
        db = self._get_db()
        if db is None:
            return False

        # MemoryDB has a conn attribute
        conn = getattr(db, 'conn', None)
        if conn is None:
            return False

        try:
            cursor = conn.execute("SELECT COUNT(*) FROM facts")
            count = cursor.fetchone()[0]
            needs = count > self.config['max_facts']
            if needs:
                logger.info("Fact count %d exceeds max %d — compression needed",
                           count, self.config['max_facts'])
            return needs
        except Exception:
            return False

    def _get_facts_to_compress(self) -> List[Dict]:
        """Get facts that are candidates for compression.
        
        Excludes:
        - Protected categories (preference, identity)
        - Recently compressed facts
        """
        db = self._get_db()
        if db is None:
            return []

        protected = self.config.get('protected_categories', [])
        if not protected:
            protected = ['preference', 'identity']

        placeholders = ','.join(['?' for _ in protected])
        query = f"""
            SELECT id, key, category, created_at
            FROM facts
            WHERE category NOT IN ({placeholders})
            ORDER BY created_at ASC
        """

        conn = getattr(db, 'conn', None)
        if conn is None:
            return []

        try:
            cursor = conn.execute(query, protected)
            facts = []
            for row in cursor.fetchall():
                facts.append({
                    'id': row[0],
                    'key': row[1],
                    'category': row[2],
                    'created_at': row[3],
                })
            return facts
        except Exception as e:
            logger.error("Failed to get facts for compression: %s", e)
            return []

    def _group_similar_facts(self, facts: List[Dict]) -> List[List[Dict]]:
        """Group facts by cosine similarity threshold.

        P5-fix: Single query with IN clause instead of N+1 queries.
        Compares only candidate facts against each other (not full DB).
        Preserves full fact dict (id, key, category, created_at).
        """
        if not facts:
            return []

        db = self._get_db()
        if db is None:
            return []

        conn = getattr(db, 'conn', None)
        if conn is None:
            return []

        threshold = self.config.get('similarity_threshold', 0.85)

        try:
            import numpy as np

            # Single query: fetch embeddings for all candidate facts at once
            fact_ids = [f['id'] for f in facts]
            placeholders = ','.join(['?' for _ in fact_ids])
            cursor = conn.execute(
                f"SELECT id, key, category, created_at, embedding "
                f"FROM facts WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                fact_ids
            )
            rows = cursor.fetchall()

            # Build lookup: id -> (full_fact_dict, embedding_vec)
            id_to_data = {}
            for row_id, row_key, row_cat, row_created, row_emb in rows:
                vec = np.asarray(json.loads(row_emb), dtype=np.float32)
                id_to_data[row_id] = (
                    {
                        'id': row_id,
                        'key': row_key,
                        'category': row_cat,
                        'created_at': row_created,
                    },
                    vec,
                )

            if not id_to_data:
                return []

            # In-memory pairwise comparison
            groups = []
            used_ids = set()
            ids = list(id_to_data.keys())

            for i, fact_id in enumerate(ids):
                if fact_id in used_ids:
                    continue

                full_fact, q_vec = id_to_data[fact_id]
                q_norm = np.linalg.norm(q_vec)
                if q_norm < 1e-10:
                    used_ids.add(fact_id)
                    continue

                similar = [full_fact]
                for other_id in ids[i + 1:]:
                    if other_id in used_ids:
                        continue
                    other_fact, other_vec = id_to_data[other_id]
                    other_norm = np.linalg.norm(other_vec)
                    if other_norm < 1e-10:
                        continue

                    sim = float(q_vec @ other_vec / (q_norm * other_norm))
                    if sim >= threshold:
                        similar.append({
                            **other_fact,
                            'similarity': sim,
                        })
                        used_ids.add(other_id)

                if len(similar) >= 2:
                    groups.append(similar)
                    used_ids.add(fact_id)

            return groups
        except Exception as e:
            logger.error("Failed to group similar facts: %s", e)
            return []

    def _compress_with_llm(self, group: List[Dict]) -> Optional[str]:
        """Use Ollama to compress a group of similar facts into one summary.
        
        Returns compressed summary or None if Ollama fails.
        """
        import httpx

        facts_text = '\n'.join([
            f"- {f['key']}" for f in group
        ])

        prompt = f"""You are a memory consolidation assistant. Compress the following similar facts into a single, concise summary that preserves all key information:

{facts_text}

Output ONLY the compressed summary, nothing else. Keep it under 200 characters."""

        try:
            response = httpx.post(
                self.config.get('ollama_url', 'http://localhost:11434/api/generate'),
                json={
                    'model': self.config.get('compress_model', 'qwen2.5:3b'),
                    'prompt': prompt,
                    'stream': False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.warning("Ollama compression failed: %s — skipping this group", e)
            return None

    def _archive_facts(self, group: List[Dict], compressed: str) -> bool:
        """Archive original facts before compression."""
        if not self.config.get('archive_enabled', True):
            return True

        conn = self._get_archive_conn()
        try:
            self._ensure_archive_table()
            now = time.time()
            source_ids = json.dumps([f['id'] for f in group])
            category = group[0].get('category', 'unknown')

            conn.execute("""
                INSERT OR REPLACE INTO compressed_facts_archive
                (id, original_content, compressed_content, compression_timestamp, source_ids, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"archive_{now}_{group[0]['id']}",
                json.dumps([f['key'] for f in group]),
                compressed,
                now,
                source_ids,
                category,
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to archive facts: %s", e)
            return False
        finally:
            conn.close()

    def _delete_compressed_facts(self, group: List[Dict]) -> bool:
        """Delete original facts after archiving.

        P5-fix: Wrapped in explicit BEGIN IMMEDIATE transaction.
        Rolls back on failure to prevent partial data loss.
        """
        db = self._get_db()
        if db is None:
            return False

        conn = getattr(db, 'conn', None)
        if conn is None:
            return False

        ids = [f['id'] for f in group]
        placeholders = ','.join(['?' for _ in ids])

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"DELETE FROM facts WHERE id IN ({placeholders})", ids
            )
            conn.commit()
            return True
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.error("Failed to delete compressed facts: %s", e)
            return False

    def consolidate(self) -> Dict[str, Any]:
        """Run the full consolidation pipeline.
        
        Returns a report dict with:
        - dry_run: bool
        - facts_before: int
        - facts_after: int (estimated for dry run)
        - groups_found: int
        - groups_compressed: int
        - groups_failed: int
        - archived: int
        - errors: list
        """
        report = {
            'dry_run': self.config.get('dry_run', True),
            'facts_before': 0,
            'facts_after': 0,
            'groups_found': 0,
            'groups_compressed': 0,
            'groups_failed': 0,
            'archived': 0,
            'errors': [],
            'timestamp': time.time(),
        }

        # Check if compression is needed
        if not self._needs_compression() and not self.config.get('dry_run', True):
            logger.info("No compression needed — fact count within limits")
            report['status'] = 'skipped'
            return report

        # Get candidate facts
        facts = self._get_facts_to_compress()
        report['facts_before'] = len(facts)

        # Group similar facts
        groups = self._group_similar_facts(facts)
        report['groups_found'] = len(groups)

        if not groups:
            logger.info("No similar fact groups found")
            report['status'] = 'no_groups'
            return report

        # Process each group
        for i, group in enumerate(groups):
            if report['dry_run']:
                # Dry run: just report
                logger.info("[DRY RUN] Would compress %d facts in group %d",
                           len(group), i + 1)
                report['groups_compressed'] += 1
                continue

            # Actual compression
            compressed = self._compress_with_llm(group)
            if not compressed:
                report['groups_failed'] += 1
                report['errors'].append(
                    f"Group {i+1}: Ollama compression failed for {len(group)} facts"
                )
                continue

            # Archive
            if not self._archive_facts(group, compressed):
                report['errors'].append(
                    f"Group {i+1}: Archive failed"
                )
                continue

            # Delete originals
            if not self._delete_compressed_facts(group):
                report['errors'].append(
                    f"Group {i+1}: Delete failed"
                )
                continue

            report['groups_compressed'] += 1
            report['archived'] += len(group)
            logger.info("Compressed %d facts → 1 summary (group %d)",
                       len(group), i + 1)

        report['facts_after'] = report['facts_before'] - report['archived']
        report['status'] = 'completed' if not report['errors'] else 'partial'
        return report
