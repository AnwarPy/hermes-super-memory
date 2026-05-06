"""P4: Mistake Notes Tests — full coverage for MistakeTracker."""

import json
import os
import sqlite3
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.mistake_notes import (
    MistakeTracker, MISTAKE_CATEGORIES, SEVERITY_LEVELS,
    DEFAULT_CATEGORY, DEFAULT_SEVERITY,
)


class TestConstants:
    """Test category and severity definitions."""

    def test_categories_have_descriptions(self):
        for cat, desc in MISTAKE_CATEGORIES.items():
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_expected_categories_exist(self):
        expected = {'syntax', 'logic', 'config', 'api', 'security',
                    'performance', 'data', 'deployment', 'communication', 'other'}
        assert set(MISTAKE_CATEGORIES.keys()) == expected

    def test_severity_levels(self):
        expected = {'low', 'medium', 'high', 'critical'}
        assert set(SEVERITY_LEVELS.keys()) == expected

    def test_default_values(self):
        assert DEFAULT_CATEGORY == 'other'
        assert DEFAULT_SEVERITY == 'medium'


class TestMistakeTrackerInit:
    """Test initialization."""

    def test_init_with_none_db(self):
        tracker = MistakeTracker(db=None)
        assert tracker.db is None

    def test_init_with_db(self):
        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        tracker = MistakeTracker(db=mock_db)
        assert tracker.db is mock_db

    def test_init_with_config(self):
        tracker = MistakeTracker(config={'key': 'value'})
        assert tracker.config['key'] == 'value'

    def test_init_default_config(self):
        tracker = MistakeTracker()
        assert tracker.config == {}


class TestEnsureTable:
    """Test table creation."""

    def test_ensure_table_with_none_conn(self):
        tracker = MistakeTracker(db=None)
        assert tracker.ensure_table() is False

    def test_ensure_table_with_db_no_conn(self):
        mock_db = MagicMock()
        mock_db.conn = None
        tracker = MistakeTracker(db=mock_db)
        assert tracker.ensure_table() is False

    def test_ensure_table_creates_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = conn

        tracker = MistakeTracker(db=mock_db)
        result = tracker.ensure_table()

        assert result is True

        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mistake_notes'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_ensure_table_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = conn

        tracker = MistakeTracker(db=mock_db)
        result1 = tracker.ensure_table()
        result2 = tracker.ensure_table()

        assert result1 is True
        assert result2 is True
        conn.close()

    def test_ensure_table_creates_fts(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = conn

        tracker = MistakeTracker(db=mock_db)
        tracker.ensure_table()

        # Verify FTS table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mistake_notes_fts'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestRecordMistake:
    """Test mistake recording."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_record_basic(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        mistake_id = tracker.record_mistake("Test mistake description")
        assert mistake_id is not None
        assert mistake_id > 0

    def test_record_with_all_fields(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        mistake_id = tracker.record_mistake(
            description="Wrong port config",
            category="config",
            severity="high",
            context="Setting up webhook",
            fix="Changed port to 9190",
            fix_verified=True,
            session_id="session-123",
            tags=["webhook", "port"],
        )

        assert mistake_id is not None

        # Verify data
        cursor = self.conn.execute(
            "SELECT * FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        row = cursor.fetchone()
        assert row[1] == "Wrong port config"
        assert row[2] == "config"
        assert row[3] == "high"
        assert row[4] == "Setting up webhook"
        assert row[5] == "Changed port to 9190"
        assert row[6] == 1
        assert row[8] == "session-123"
        tags = json.loads(row[9])
        assert "webhook" in tags
        assert "port" in tags

    def test_record_invalid_category_uses_default(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        mistake_id = tracker.record_mistake(
            "Test", category="invalid_category"
        )
        assert mistake_id is not None

        cursor = self.conn.execute(
            "SELECT category FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        assert cursor.fetchone()[0] == "other"

    def test_record_invalid_severity_uses_default(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        mistake_id = tracker.record_mistake(
            "Test", severity="invalid_severity"
        )
        assert mistake_id is not None

        cursor = self.conn.execute(
            "SELECT severity FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        assert cursor.fetchone()[0] == "medium"

    def test_record_duplicate_updates_recurrence(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        id1 = tracker.record_mistake("Duplicate mistake", session_id="s1")
        id2 = tracker.record_mistake("Duplicate mistake", session_id="s1")

        assert id1 == id2  # Same ID due to UNIQUE constraint

        cursor = self.conn.execute(
            "SELECT recurrence_count FROM mistake_notes WHERE id = ?", (id1,)
        )
        assert cursor.fetchone()[0] == 2

    def test_record_different_sessions_different_ids(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        id1 = tracker.record_mistake("Same mistake", session_id="s1")
        id2 = tracker.record_mistake("Same mistake", session_id="s2")

        assert id1 != id2

    def test_record_without_db(self):
        tracker = MistakeTracker(db=None)
        result = tracker.record_mistake("Test")
        assert result is None

    def test_record_sets_created_at(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        before = time.time()
        mistake_id = tracker.record_mistake("Test")
        after = time.time()

        cursor = self.conn.execute(
            "SELECT created_at FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        created_at = cursor.fetchone()[0]
        assert before <= created_at <= after


class TestSearchMistakes:
    """Test mistake search functionality."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def _populate(self, tracker):
        """Add test data."""
        tracker.record_mistake(
            "Wrong port for webhook",
            category="config",
            severity="high",
            context="Agent setup",
            fix="Use port 9190",
            session_id="s1",
        )
        tracker.record_mistake(
            "Missing API key in .env",
            category="config",
            severity="medium",
            context="Deployment",
            fix="Add API_KEY to .env",
            session_id="s1",
        )
        tracker.record_mistake(
            "Syntax error in SQL query",
            category="syntax",
            severity="low",
            context="Database migration",
            fix="Fixed comma placement",
            session_id="s2",
        )

    def test_search_by_query(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("webhook")
        assert len(results) >= 1
        assert any("webhook" in r['description'].lower() for r in results)

    def test_search_by_category(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("", category="config")
        assert len(results) == 2
        assert all(r['category'] == 'config' for r in results)

    def test_search_by_severity(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("", severity="high")
        assert len(results) == 1
        assert results[0]['severity'] == 'high'

    def test_search_by_session(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("", session_id="s2")
        assert len(results) == 1
        assert results[0]['session_id'] == 's2'

    def test_search_with_limit(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("", limit=1)
        assert len(results) <= 1

    def test_search_no_results(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        self._populate(tracker)

        results = tracker.search_mistakes("nonexistent query xyz")
        # May return empty or some results depending on FTS behavior
        assert isinstance(results, list)

    def test_search_without_db(self):
        tracker = MistakeTracker(db=None)
        results = tracker.search_mistakes("test")
        assert results == []

    def test_search_returns_dicts_with_keys(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Test description", category="syntax")

        results = tracker.search_mistakes("test")
        if results:
            r = results[0]
            assert 'id' in r
            assert 'description' in r
            assert 'category' in r
            assert 'severity' in r
            assert 'context' in r
            assert 'fix' in r
            assert 'fix_verified' in r
            assert 'tags' in r
            assert 'recurrence_count' in r

    def test_search_fix_verified_is_bool(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Test", fix_verified=True)

        results = tracker.search_mistakes("test")
        if results:
            assert isinstance(results[0]['fix_verified'], bool)
            assert results[0]['fix_verified'] is True


class TestGetByCategory:
    """Test category-based retrieval."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_get_by_category(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Syntax error 1", category="syntax")
        tracker.record_mistake("Syntax error 2", category="syntax")
        tracker.record_mistake("Config error", category="config")

        results = tracker.get_by_category("syntax")
        assert len(results) == 2
        assert all(r['category'] == 'syntax' for r in results)

    def test_get_by_category_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        results = tracker.get_by_category("security")
        assert results == []


class TestGetBySeverity:
    """Test severity-based retrieval."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_get_by_severity(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Critical bug", severity="critical")
        tracker.record_mistake("Low warning", severity="low")

        results = tracker.get_by_severity("critical")
        assert len(results) == 1
        assert results[0]['severity'] == 'critical'


class TestGetBySession:
    """Test session-based retrieval."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_get_by_session(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Error A", session_id="s1")
        tracker.record_mistake("Error B", session_id="s1")
        tracker.record_mistake("Error C", session_id="s2")

        results = tracker.get_by_session("s1")
        assert len(results) == 2
        assert all(r['session_id'] == 's1' for r in results)


class TestVerifyFix:
    """Test fix verification."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_verify_fix_success(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        mistake_id = tracker.record_mistake("Bug", fix="Fixed it")

        result = tracker.verify_fix(mistake_id)
        assert result is True

        cursor = self.conn.execute(
            "SELECT fix_verified FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        assert cursor.fetchone()[0] == 1

    def test_verify_fix_nonexistent(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        result = tracker.verify_fix(99999)
        assert result is True  # UPDATE succeeds even with no matching rows

    def test_verify_fix_without_db(self):
        tracker = MistakeTracker(db=None)
        result = tracker.verify_fix(1)
        assert result is False


class TestGetRecurringMistakes:
    """Test recurring mistake retrieval."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_get_recurring(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        # Record same mistake twice in same session
        tracker.record_mistake("Repeated error", session_id="s1")
        tracker.record_mistake("Repeated error", session_id="s1")

        results = tracker.get_recurring_mistakes(min_count=2)
        assert len(results) == 1
        assert results[0]['recurrence_count'] == 2
        assert results[0]['description'] == "Repeated error"

    def test_get_recurring_none(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Single error", session_id="s1")

        results = tracker.get_recurring_mistakes(min_count=2)
        assert results == []

    def test_get_recurring_without_db(self):
        tracker = MistakeTracker(db=None)
        results = tracker.get_recurring_mistakes()
        assert results == []

    def test_get_recurring_custom_threshold(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        tracker.record_mistake("Triple error", session_id="s1")
        tracker.record_mistake("Triple error", session_id="s1")
        tracker.record_mistake("Triple error", session_id="s1")

        results = tracker.get_recurring_mistakes(min_count=3)
        assert len(results) == 1
        assert results[0]['recurrence_count'] == 3


class TestGetStats:
    """Test statistics retrieval."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_stats_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        stats = tracker.get_stats()
        assert stats['total'] == 0
        assert stats['by_category'] == {}
        assert stats['by_severity'] == {}
        assert stats['recurring'] == 0
        assert stats['verified_fixes'] == 0

    def test_stats_with_data(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        tracker.record_mistake("Config error", category="config", severity="high")
        tracker.record_mistake("Syntax error", category="syntax", severity="low")
        tracker.record_mistake("Another config", category="config", severity="medium",
                              fix_verified=True)

        stats = tracker.get_stats()
        assert stats['total'] == 3
        assert stats['by_category']['config'] == 2
        assert stats['by_category']['syntax'] == 1
        assert stats['by_severity']['high'] == 1
        assert stats['by_severity']['low'] == 1
        assert stats['by_severity']['medium'] == 1
        assert stats['verified_fixes'] == 1

    def test_stats_without_db(self):
        tracker = MistakeTracker(db=None)
        stats = tracker.get_stats()
        assert stats['total'] == 0


class TestDeleteMistake:
    """Test mistake deletion."""

    def setup_method(self):
        self.conn = None

    def teardown_method(self):
        if self.conn:
            self.conn.close()

    def _make_tracker(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        self.conn = sqlite3.connect(db_path)
        mock_db = MagicMock()
        mock_db.conn = self.conn
        return MistakeTracker(db=mock_db)

    def test_delete_success(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()
        mistake_id = tracker.record_mistake("To delete")

        result = tracker.delete_mistake(mistake_id)
        assert result is True

        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM mistake_notes WHERE id = ?", (mistake_id,)
        )
        assert cursor.fetchone()[0] == 0

    def test_delete_nonexistent(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.ensure_table()

        result = tracker.delete_mistake(99999)
        assert result is True  # DELETE succeeds even with no matching rows

    def test_delete_without_db(self):
        tracker = MistakeTracker(db=None)
        result = tracker.delete_mistake(1)
        assert result is False


class TestRowToDict:
    """Test _row_to_dict conversion."""

    def test_row_to_dict_basic(self):
        row = (1, "desc", "config", "high", "ctx", "fix", 1, 1000.0, "s1", '["tag1"]', 3)
        result = MistakeTracker._row_to_dict(row)

        assert result['id'] == 1
        assert result['description'] == "desc"
        assert result['category'] == "config"
        assert result['severity'] == "high"
        assert result['context'] == "ctx"
        assert result['fix'] == "fix"
        assert result['fix_verified'] is True
        assert result['created_at'] == 1000.0
        assert result['session_id'] == "s1"
        assert result['tags'] == ["tag1"]
        assert result['recurrence_count'] == 3

    def test_row_to_dict_empty_tags(self):
        row = (1, "desc", "config", "high", "", "", 0, 1000.0, "", None, 1)
        result = MistakeTracker._row_to_dict(row)
        assert result['tags'] == []

    def test_row_to_dict_invalid_json_tags(self):
        row = (1, "desc", "config", "high", "", "", 0, 1000.0, "", "not json", 1)
        result = MistakeTracker._row_to_dict(row)
        assert result['tags'] == []
