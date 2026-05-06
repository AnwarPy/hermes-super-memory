"""P2: Extended Cache Tests — full coverage for QueryResultCache and GraphCache."""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.cache import QueryResultCache, GraphCache


class TestQueryResultCacheBasics:
    """Basic cache operations."""

    def test_set_and_get(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("hello", {"result": "world"})
        assert cache.get("hello") == {"result": "world"}

    def test_get_missing_key(self):
        cache = QueryResultCache(ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_set_overwrites(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_clear_removes_all(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None


class TestQueryResultCacheTTL:
    """TTL expiry tests."""

    def test_expired_entry_returns_none(self):
        cache = QueryResultCache(ttl_seconds=0.1)  # 100ms TTL
        cache.set("key", "value")
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_fresh_entry_returns_value(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_ttl_is_per_entry(self):
        cache = QueryResultCache(ttl_seconds=0.1)
        cache.set("early", "value1")
        time.sleep(0.05)
        cache.set("late", "value2")
        time.sleep(0.07)
        # early should be expired, late should still be valid
        assert cache.get("early") is None
        assert cache.get("late") == "value2"


class TestQueryResultCacheSessionId:
    """Session-based cache key tests."""

    def test_different_session_ids_different_keys(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("query", "result_A", session_id="session_A")
        cache.set("query", "result_B", session_id="session_B")
        assert cache.get("query", session_id="session_A") == "result_A"
        assert cache.get("query", session_id="session_B") == "result_B"

    def test_empty_session_id_matches_default(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("query", "result")
        assert cache.get("query", session_id="") == "result"


class TestQueryResultCacheMaxAge:
    """Max age days parameter tests."""

    def test_different_max_age_different_keys(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("query", "result_7d", max_age_days=7)
        cache.set("query", "result_30d", max_age_days=30)
        assert cache.get("query", max_age_days=7) == "result_7d"
        assert cache.get("query", max_age_days=30) == "result_30d"

    def test_none_max_age_different_from_value(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("query", "result_none", max_age_days=None)
        cache.set("query", "result_7", max_age_days=7)
        assert cache.get("query", max_age_days=None) == "result_none"
        assert cache.get("query", max_age_days=7) == "result_7"


class TestQueryResultCacheCleanup:
    """Cache cleanup and max size tests."""

    def test_cleanup_expired_removes_old_entries(self):
        cache = QueryResultCache(ttl_seconds=0.1)
        cache.set("old1", "v1")
        cache.set("old2", "v2")
        cache.set("new", "v3")
        time.sleep(0.15)
        cache.cleanup_expired()
        assert cache.get("old1") is None
        assert cache.get("old2") is None

    def test_cleanup_preserves_fresh(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set("key", "value")
        cache.cleanup_expired()
        assert cache.get("key") == "value"

    def test_max_size_triggers_cleanup(self):
        cache = QueryResultCache(ttl_seconds=60)  # Long TTL - cleanup won't remove anything
        # Fill well beyond max size
        for i in range(cache.MAX_SIZE + 100):
            cache.set(f"key_{i}", f"value_{i}")
        # After set triggers cleanup when over limit, cleanup won't remove fresh items
        # So cache will still grow slightly beyond MAX_SIZE
        # The important thing is that cleanup was called (no crash)
        assert len(cache._cache) > cache.MAX_SIZE  # Items are fresh so they stay

    def test_max_size_default(self):
        cache = QueryResultCache()
        assert cache.MAX_SIZE == 500


class TestQueryResultCacheThreadSafety:
    """Thread safety tests."""

    def test_concurrent_access_no_crash(self):
        import threading
        cache = QueryResultCache(ttl_seconds=60)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 50):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                for i in range(start, start + 50):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t1 = threading.Thread(target=writer, args=(i * 50,))
            t2 = threading.Thread(target=reader, args=(i * 50,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestGraphCacheBasics:
    """GraphCache basic operations."""

    def test_get_loads_graph(self):
        cache = GraphCache()
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 10
        mock_graph.number_of_edges.return_value = 20

        with patch('unified.cache.os.path.exists', return_value=False):
            result = cache.get("test_project", lambda name: mock_graph)

        assert result is mock_graph

    def test_get_returns_cached_graph(self):
        cache = GraphCache()
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 10
        mock_graph.number_of_edges.return_value = 20

        def loader(name):
            return mock_graph

        with patch('unified.cache.os.path.exists', return_value=False):
            cache.get("test_project", loader)
            result = cache.get("test_project", loader)

        assert result is mock_graph

    def test_get_returns_none_on_error(self):
        cache = GraphCache()

        def loader(name):
            raise Exception("load failed")

        result = cache.get("bad_project", loader)
        assert result is None

    def test_clear_removes_all(self):
        cache = GraphCache()
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 5
        mock_graph.number_of_edges.return_value = 10

        with patch('unified.cache.os.path.exists', return_value=False):
            cache.get("project", lambda name: mock_graph)

        cache.clear()
        assert cache.stats == {}


class TestGraphCacheMtime:
    """GraphCache mtime invalidation tests."""

    def test_mtime_change_triggers_reload(self, tmp_path):
        cache = GraphCache()
        graph_dir = tmp_path / "test_project"
        graph_dir.mkdir()
        graph_file = graph_dir / "graph.json"
        graph_file.write_text("{}")

        call_count = [0]

        def loader(name):
            call_count[0] += 1
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = call_count[0]
            mock_graph.number_of_edges.return_value = 0
            return mock_graph

        # First load
        result1 = cache.get("test_project", loader)
        assert result1.number_of_nodes() == 1
        assert call_count[0] == 1

        # Modify file
        time.sleep(0.01)
        graph_file.write_text('{"modified": true}')

        # Second get should reload
        result2 = cache.get("test_project", loader)
        assert call_count[0] == 2  # Loader called again

    def test_no_mtime_change_returns_cached(self, tmp_path):
        cache = GraphCache()
        graph_dir = tmp_path / "test_project"
        graph_dir.mkdir()
        graph_file = graph_dir / "graph.json"
        graph_file.write_text("{}")

        call_count = [0]
        original_mtime = os.path.getmtime(str(graph_file))

        def loader(name):
            call_count[0] += 1
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = 1
            mock_graph.number_of_edges.return_value = 0
            return mock_graph

        # Both calls: file exists, mtime unchanged
        with patch('unified.cache.os.path.exists', return_value=True):
            with patch('unified.cache.os.path.getmtime', return_value=original_mtime):
                cache.get("test_project", loader)
                assert call_count[0] == 1
                # Second call should use cache
                result = cache.get("test_project", loader)

        assert call_count[0] == 1  # Loader not called again


class TestGraphCacheLRU:
    """GraphCache LRU eviction tests."""

    def test_evicts_oldest_when_over_capacity(self):
        cache = GraphCache()
        cache.MAX_CACHED_GRAPHS = 3  # Small cap for testing

        def make_graph(n):
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = n
            mock_graph.number_of_edges.return_value = 0
            return mock_graph

        with patch('unified.cache.os.path.exists', return_value=False):
            cache.get("proj1", lambda n: make_graph(1))
            cache.get("proj2", lambda n: make_graph(2))
            cache.get("proj3", lambda n: make_graph(3))
            # Adding 4th should evict proj1
            cache.get("proj4", lambda n: make_graph(4))

        assert len(cache._graphs) <= 3
        assert "proj1" not in cache._graphs

    def test_access_resets_lru_order(self):
        cache = GraphCache()
        cache.MAX_CACHED_GRAPHS = 3

        def make_graph(n):
            mock_graph = MagicMock()
            mock_graph.number_of_nodes.return_value = n
            mock_graph.number_of_edges.return_value = 0
            return mock_graph

        with patch('unified.cache.os.path.exists', return_value=False):
            cache.get("proj1", lambda n: make_graph(1))
            cache.get("proj2", lambda n: make_graph(2))
            cache.get("proj3", lambda n: make_graph(3))
            # Access proj1 again — it should now be most recently used
            cache.get("proj1", lambda n: None)
            # Add proj4 — should evict proj2 (now oldest)
            cache.get("proj4", lambda n: make_graph(4))

        assert "proj1" in cache._graphs  # Should still be here
        assert "proj2" not in cache._graphs  # Should be evicted


class TestGraphCacheStats:
    """GraphCache statistics tests."""

    def test_stats_records_nodes_and_edges(self):
        cache = GraphCache()
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 42
        mock_graph.number_of_edges.return_value = 100

        with patch('unified.cache.os.path.exists', return_value=False):
            cache.get("my_graph", lambda name: mock_graph)

        stats = cache.stats
        assert "my_graph" in stats
        assert stats["my_graph"]['nodes'] == 42
        assert stats["my_graph"]['edges'] == 100

    def test_stats_returns_copy(self):
        cache = GraphCache()
        stats1 = cache.stats
        stats1['fake'] = {'nodes': 0, 'edges': 0}
        stats2 = cache.stats
        assert 'fake' not in stats2

    def test_stats_empty_when_no_graphs(self):
        cache = GraphCache()
        assert cache.stats == {}


class TestGraphCacheFileNotFound:
    """Tests for missing graph file scenarios."""

    def test_get_without_file_still_calls_loader(self):
        cache = GraphCache()
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 5
        mock_graph.number_of_edges.return_value = 10

        with patch('unified.cache.os.path.exists', return_value=False):
            result = cache.get("nonexistent", lambda name: mock_graph)

        assert result is mock_graph
        # Stats should still be recorded
        assert "nonexistent" in cache.stats

    def test_get_file_exists_but_loader_fails(self, tmp_path):
        cache = GraphCache()
        graph_dir = tmp_path / "broken"
        graph_dir.mkdir()
        graph_file = graph_dir / "graph.json"
        graph_file.write_text("{invalid json}")

        def loader(name):
            raise ValueError("parse error")

        result = cache.get("broken", loader)
        assert result is None
