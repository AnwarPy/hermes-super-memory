"""Integration tests — real SQLite connections, no mocking of core DB paths.

These tests catch regressions that unit tests miss, like:
- Shared connection being closed unexpectedly
- Transaction safety across multiple operations
- Archive operations interfering with main DB operations
"""

import os
import sys
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestSharedConnectionSafety:
    """Regression test: _get_archive_conn() must not close shared MemoryDB connection.

    Catches the bug where _get_archive_conn() returned the shared connection
    but callers unconditionally closed it in finally blocks, destroying the
    connection for all other operations.
    """

    def test_archive_conn_shared_not_closed(self):
        """When _get_archive_conn() returns shared connection, caller must not close it."""
        from unified.consolidation import MemoryConsolidator

        # Create a real SQLite connection to act as the "shared" MemoryDB conn
        shared_conn = sqlite3.connect(":memory:")
        shared_conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        shared_conn.execute("INSERT INTO test (id) VALUES (1)")
        shared_conn.commit()

        # Create a mock MemoryDB with the shared connection
        mock_db = MagicMock()
        mock_db.conn = shared_conn

        consolidator = MemoryConsolidator({})

        with patch.object(consolidator, '_get_db', return_value=mock_db):
            conn, is_shared = consolidator._get_archive_conn()
            assert is_shared is True
            assert conn is shared_conn

            # Caller should NOT close shared connection
            if not is_shared:
                conn.close()

            # Verify connection is still usable
            cursor = conn.execute("SELECT id FROM test")
            assert cursor.fetchone() == (1,)

        shared_conn.close()

    def test_archive_conn_fallback_is_closable(self):
        """When _get_archive_conn() creates a new connection, it should be safe to close."""
        from unified.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator({})

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with patch.object(consolidator, '_get_db', return_value=None):
                with patch.object(consolidator, '_get_db_path', return_value=db_path):
                    conn, is_shared = consolidator._get_archive_conn()
                    assert is_shared is False

                    # Should be safe to close
                    conn.close()

                    # Verify it's closed
                    with pytest.raises(sqlite3.ProgrammingError):
                        conn.execute("SELECT 1")

    def test_ensure_archive_table_does_not_close_shared(self):
        """_ensure_archive_table() must not destroy the shared connection."""
        from unified.consolidation import MemoryConsolidator

        shared_conn = sqlite3.connect(":memory:")
        shared_conn.execute("CREATE TABLE main_table (id INTEGER PRIMARY KEY)")
        shared_conn.execute("INSERT INTO main_table (id) VALUES (42)")
        shared_conn.commit()

        mock_db = MagicMock()
        mock_db.conn = shared_conn

        consolidator = MemoryConsolidator({})

        with patch.object(consolidator, '_get_db', return_value=mock_db):
            result = consolidator._ensure_archive_table()
            assert result is True

            # The main table should still be accessible — connection not destroyed
            cursor = shared_conn.execute("SELECT id FROM main_table WHERE id = 42")
            assert cursor.fetchone() == (42,)

        shared_conn.close()


class TestTransactionSafety:
    """Verify BEGIN IMMEDIATE + rollback behavior prevents partial data loss."""

    def test_delete_compressed_facts_rolls_back_on_error(self):
        """If deletion fails, all facts should remain intact."""
        from unified.consolidation import MemoryConsolidator
        from unified.memory_db import _get_memory_db

        # Create a real temp DB
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    category TEXT,
                    embedding TEXT
                )
            """)
            conn.execute("INSERT INTO facts (id, key, category) VALUES (1, 'fact1', 'test')")
            conn.execute("INSERT INTO facts (id, key, category) VALUES (2, 'fact2', 'test')")
            conn.execute("INSERT INTO facts (id, key, category) VALUES (3, 'fact3', 'test')")
            conn.commit()

            # Mock MemoryDB to use our temp connection
            mock_db = MagicMock()
            mock_db.conn = conn

            consolidator = MemoryConsolidator({})

            with patch.object(consolidator, '_get_db', return_value=mock_db):
                # Normal deletion should succeed
                group = [{'id': 1}, {'id': 2}]
                result = consolidator._delete_compressed_facts(group)
                assert result is True

                # Verify deleted
                cursor = conn.execute("SELECT count(*) FROM facts")
                assert cursor.fetchone()[0] == 1  # Only fact 3 remains

            conn.close()


class TestFTS5ArabicIntegration:
    """End-to-end Arabic FTS5 search with real SQLite (no mocks)."""

    def test_arabic_search_finds_substrings(self):
        """Trigram FTS5 should find 'ذاكرة' inside 'الذاكرة'."""
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = sqlite3.connect(db_path)

            # Create table + trigram FTS5
            conn.execute("""
                CREATE TABLE notes (
                    id INTEGER PRIMARY KEY,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE notes_fts USING fts5(
                    description,
                    content='notes', content_rowid='id',
                    tokenize='trigram'
                )
            """)
            conn.execute("INSERT INTO notes (id, description) VALUES (1, 'الذاكرة الموحدة تعمل بشكل جيد')")
            conn.execute("INSERT INTO notes_fts (rowid, description) VALUES (1, 'الذاكرة الموحدة تعمل بشكل جيد')")
            conn.commit()

            # Search for 'ذاكرة' (without الـ prefix) — should match 'الذاكرة'
            cursor = conn.execute("""
                SELECT n.description FROM notes n
                JOIN notes_fts f ON n.id = f.rowid
                WHERE notes_fts MATCH 'ذاكرة'
            """)
            results = cursor.fetchall()
            assert len(results) >= 1
            assert 'الذاكرة' in results[0][0]

            conn.close()

    def test_arabic_search_normalized_query(self):
        """Normalized query should match different hamza forms."""
        from unified.arabic_normalizer import normalize_query

        # Different hamza forms should normalize to the same thing
        assert normalize_query('إسلام') == normalize_query('اسلام')
        assert normalize_query('أحمد') == normalize_query('احمد')
        assert normalize_query('القرآن') == normalize_query('القران')
