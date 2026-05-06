"""P4: Mistake Notes — تسجيل وتصنيف الأخطاء لإعادة التمثيل.

This module provides a structured way to record, classify, and retrieve
mistake notes. Mistakes are stored separately from regular facts to enable:
1. Learning from past errors (avoiding repetition)
2. Categorizing mistakes by type (syntax, logic, config, etc.)
3. Tracking fix attempts and outcomes
4. Searching mistakes alongside regular memory

Usage:
    from unified.mistake_notes import MistakeTracker

    tracker = MistakeTracker(db)

    # Record a mistake
    tracker.record_mistake(
        description="Used wrong port for webhook (9090 instead of 9190)",
        category="config",
        severity="medium",
        context="Setting up Telegram webhook for agent-5",
        fix="Changed WEBHOOK_PORT to 9190 in config.yaml",
    )

    # Search mistakes
    results = tracker.search_mistakes(query="webhook port")

    # Get mistakes by category
    config_errors = tracker.get_by_category("config")
"""

import json
import logging
import sqlite3
import time
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ============================================================
# Mistake categories and severity levels
# ============================================================

MISTAKE_CATEGORIES = {
    'syntax': 'خطأ في صيغة الكود أو الأمر',
    'logic': 'خطأ منطقي في الخوارزمية أو التدفق',
    'config': 'خطأ في الإعدادات أو التكوين',
    'api': 'خطأ في استدعاء API أو التكامل',
    'security': 'خطأ أمني أو ثغرة',
    'performance': 'مشكلة أداء أو استهلاك موارد',
    'data': 'خطأ في معالجة البيانات أو التنسيق',
    'deployment': 'خطأ في النشر أو البيئة',
    'communication': 'سوء فهم أو تواصل',
    'other': 'أخرى',
}

SEVERITY_LEVELS = {
    'low': 'تأثير محدود، سهل الإصلاح',
    'medium': 'تأثير متوسط، يحتاج تدخل',
    'high': 'تأثير كبير، قد يعطل النظام',
    'critical': 'حرج، يتطلب تدخل فوري',
}

DEFAULT_SEVERITY = 'medium'
DEFAULT_CATEGORY = 'other'

# ============================================================
# SQL schema
# ============================================================

MISTAKE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mistake_notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    description     TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'other',
    severity        TEXT NOT NULL DEFAULT 'medium',
    context         TEXT DEFAULT '',
    fix             TEXT DEFAULT '',
    fix_verified    INTEGER DEFAULT 0,
    created_at      REAL NOT NULL,
    session_id      TEXT DEFAULT '',
    tags            TEXT DEFAULT '[]',
    related_fact_id INTEGER REFERENCES facts(id),
    recurrence_count INTEGER DEFAULT 1,
    UNIQUE(description, session_id)
)
"""

MISTAKE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS mistake_notes_fts USING fts5(
    description, context, fix, tags,
    content='mistake_notes', content_rowid='id'
)
"""


class MistakeTracker:
    """P4: Tracks and manages mistake notes."""

    def __init__(self, db=None, config: Optional[Dict] = None):
        """Initialize mistake tracker.

        Args:
            db: MemoryDB connection (has .conn attribute)
            config: Optional configuration overrides
        """
        self.db = db
        self.config = config or {}
        self._conn = None

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        """Get SQLite connection from MemoryDB."""
        if self._conn is not None:
            return self._conn

        if self.db is None:
            return None

        conn = getattr(self.db, 'conn', None)
        if conn is None:
            return None

        self._conn = conn
        return conn

    def ensure_table(self) -> bool:
        """Create mistake_notes table if it doesn't exist."""
        conn = self._get_conn()
        if conn is None:
            return False

        try:
            conn.execute(MISTAKE_TABLE_SQL)
            conn.execute(MISTAKE_FTS_SQL)
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to create mistake_notes table: %s", e)
            return False

    def record_mistake(
        self,
        description: str,
        category: str = DEFAULT_CATEGORY,
        severity: str = DEFAULT_SEVERITY,
        context: str = "",
        fix: str = "",
        fix_verified: bool = False,
        session_id: str = "",
        tags: Optional[List[str]] = None,
        related_fact_id: Optional[int] = None,
    ) -> Optional[int]:
        """Record a new mistake note.

        Returns the mistake ID or None on failure.
        """
        conn = self._get_conn()
        if conn is None:
            return None

        if not self.ensure_table():
            return None

        # Validate category
        if category not in MISTAKE_CATEGORIES:
            logger.warning("Unknown mistake category: %s — using 'other'", category)
            category = DEFAULT_CATEGORY

        # Validate severity
        if severity not in SEVERITY_LEVELS:
            logger.warning("Unknown severity level: %s — using 'medium'", severity)
            severity = DEFAULT_SEVERITY

        tags_json = json.dumps(tags or [])
        now = time.time()

        try:
            # Check if duplicate exists
            cursor = conn.execute(
                "SELECT id, recurrence_count FROM mistake_notes WHERE description = ? AND session_id = ?",
                (description, session_id),
            )
            existing = cursor.fetchone()

            if existing:
                # Update recurrence count
                mistake_id = existing[0]
                new_count = existing[1] + 1
                conn.execute(
                    "UPDATE mistake_notes SET recurrence_count = ? WHERE id = ?",
                    (new_count, mistake_id),
                )
                conn.commit()
                logger.info("Updated mistake #%d recurrence to %d: %s",
                           mistake_id, new_count, description[:50])
            else:
                # Insert new record
                cursor = conn.execute(
                    """INSERT INTO mistake_notes
                       (description, category, severity, context, fix, fix_verified,
                        created_at, session_id, tags, related_fact_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        description,
                        category,
                        severity,
                        context,
                        fix,
                        int(fix_verified),
                        now,
                        session_id,
                        tags_json,
                        related_fact_id,
                    ),
                )
                conn.commit()
                mistake_id = cursor.lastrowid

                # Update FTS index
                try:
                    conn.execute(
                        "INSERT INTO mistake_notes_fts (rowid, description, context, fix, tags) VALUES (?, ?, ?, ?, ?)",
                        (mistake_id, description, context, fix, tags_json),
                    )
                    conn.commit()
                except Exception:
                    pass  # FTS is optional

                logger.info("Recorded mistake #%d: [%s/%s] %s",
                           mistake_id, category, severity, description[:50])

            return mistake_id

        except Exception as e:
            logger.error("Failed to record mistake: %s", e)
            return None

    def search_mistakes(
        self,
        query: str,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Search mistake notes.

        Uses FTS if available, falls back to LIKE search.
        """
        conn = self._get_conn()
        if conn is None:
            return []

        if not self.ensure_table():
            return []

        # Try FTS search first
        try:
            fts_query = query.replace(' ', ' OR ')
            sql = """
                SELECT mn.id, mn.description, mn.category, mn.severity,
                       mn.context, mn.fix, mn.fix_verified, mn.created_at,
                       mn.session_id, mn.tags, mn.recurrence_count
                FROM mistake_notes mn
                JOIN mistake_notes_fts fts ON mn.id = fts.rowid
                WHERE mistake_notes_fts MATCH ?
            """
            params = [fts_query]

            if category:
                sql += " AND mn.category = ?"
                params.append(category)
            if severity:
                sql += " AND mn.severity = ?"
                params.append(severity)
            if session_id:
                sql += " AND mn.session_id = ?"
                params.append(session_id)

            sql += " ORDER BY mn.recurrence_count DESC, mn.created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            fts_results = [self._row_to_dict(row) for row in cursor.fetchall()]

            # Fallback to LIKE if FTS returns no results (Arabic tokenization issue)
            if not fts_results and query:
                return self._search_like(query, category, severity, session_id, limit)
            return fts_results

        except Exception:
            # Fallback to LIKE search
            return self._search_like(query, category, severity, session_id, limit)

    def _search_like(
        self,
        query: str,
        category: Optional[str],
        severity: Optional[str],
        session_id: Optional[str],
        limit: int,
    ) -> List[Dict]:
        """Fallback LIKE-based search."""
        conn = self._get_conn()
        if conn is None:
            return []

        sql = """
            SELECT id, description, category, severity, context, fix,
                   fix_verified, created_at, session_id, tags, recurrence_count
            FROM mistake_notes
            WHERE (description LIKE ? OR context LIKE ? OR fix LIKE ?)
        """
        like_pattern = f"%{query}%"
        params = [like_pattern, like_pattern, like_pattern]

        if category:
            sql += " AND category = ?"
            params.append(category)
        if severity:
            sql += " AND severity = ?"
            params.append(severity)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " ORDER BY recurrence_count DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """Get all mistakes in a category."""
        return self.search_mistakes("", category=category, limit=limit)

    def get_by_severity(self, severity: str, limit: int = 50) -> List[Dict]:
        """Get all mistakes with a severity level."""
        return self.search_mistakes("", severity=severity, limit=limit)

    def get_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get all mistakes from a session."""
        return self.search_mistakes("", session_id=session_id, limit=limit)

    def verify_fix(self, mistake_id: int) -> bool:
        """Mark a fix as verified."""
        conn = self._get_conn()
        if conn is None:
            return False

        try:
            conn.execute(
                "UPDATE mistake_notes SET fix_verified = 1 WHERE id = ?",
                (mistake_id,),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to verify fix for mistake #%d: %s", mistake_id, e)
            return False

    def get_recurring_mistakes(self, min_count: int = 2) -> List[Dict]:
        """Get mistakes that have occurred multiple times."""
        conn = self._get_conn()
        if conn is None:
            return []

        try:
            cursor = conn.execute(
                """SELECT id, description, category, severity, context, fix,
                          fix_verified, created_at, session_id, tags, recurrence_count
                   FROM mistake_notes
                   WHERE recurrence_count >= ?
                   ORDER BY recurrence_count DESC""",
                (min_count,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get recurring mistakes: %s", e)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get mistake statistics."""
        conn = self._get_conn()
        if conn is None:
            return {'total': 0, 'by_category': {}, 'by_severity': {}}

        stats = {
            'total': 0,
            'by_category': {},
            'by_severity': {},
            'recurring': 0,
            'verified_fixes': 0,
        }

        try:
            cursor = conn.execute("SELECT COUNT(*) FROM mistake_notes")
            stats['total'] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT category, COUNT(*) FROM mistake_notes GROUP BY category"
            )
            stats['by_category'] = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute(
                "SELECT severity, COUNT(*) FROM mistake_notes GROUP BY severity"
            )
            stats['by_severity'] = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute(
                "SELECT COUNT(*) FROM mistake_notes WHERE recurrence_count > 1"
            )
            stats['recurring'] = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT COUNT(*) FROM mistake_notes WHERE fix_verified = 1"
            )
            stats['verified_fixes'] = cursor.fetchone()[0]

        except Exception as e:
            logger.error("Failed to get mistake stats: %s", e)

        return stats

    def delete_mistake(self, mistake_id: int) -> bool:
        """Delete a mistake note."""
        conn = self._get_conn()
        if conn is None:
            return False

        try:
            conn.execute("DELETE FROM mistake_notes WHERE id = ?", (mistake_id,))
            # FTS5 external content table: use special 'delete' command
            try:
                conn.execute(
                    "INSERT INTO mistake_notes_fts(mistake_notes_fts, rowid) VALUES('delete', ?)",
                    (mistake_id,),
                )
            except Exception:
                pass  # FTS delete is best-effort
            conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to delete mistake #%d: %s", mistake_id, e)
            return False

    @staticmethod
    def _row_to_dict(row) -> Dict:
        """Convert a database row to a dictionary."""
        tags = []
        try:
            tags = json.loads(row[9]) if row[9] else []
        except (json.JSONDecodeError, TypeError):
            pass

        return {
            'id': row[0],
            'description': row[1],
            'category': row[2],
            'severity': row[3],
            'context': row[4],
            'fix': row[5],
            'fix_verified': bool(row[6]),
            'created_at': row[7],
            'session_id': row[8],
            'tags': tags,
            'recurrence_count': row[10],
        }
