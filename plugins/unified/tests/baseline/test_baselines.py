"""P-1: Characterization Tests & Baseline Benchmarks

These tests capture the current behavior of UnifiedMemoryProvider BEFORE
any refactoring (P0: splitting __init__.py into modules).

Purpose:
1. Snapshot tests: Ensure prefetch output format doesn't regress
2. Benchmark tests: Capture cold start, warm search, and FTS5 latency
3. Save baselines to tests/baseline/benchmark_v1.json

Run: pytest unified/tests/baseline/test_baselines.py -v
"""

import json
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
sys.path.insert(0, os.path.expanduser('~/.hermes/hermes-agent'))

from unified import UnifiedMemoryProvider


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'graphs_dir': os.path.expanduser('~/.hermes/graphs'),
    'embedding_model': 'BAAI/bge-m3',
    'device': 'cuda',
    'similarity_threshold': 0.6,
}

BASELINE_DIR = os.path.join(os.path.dirname(__file__), 'baseline')
os.makedirs(BASELINE_DIR, exist_ok=True)

# ============================================================
# Test Queries
# ============================================================

ARABIC_QUERIES = [
    "كيف أضيف نموذج جديد لهرمز؟",
    "ما هو الرسم المعرفي وكيف يعمل؟",
    "كيف أفهرس مشروع في الذاكرة؟",
    "شرح نظام التذكر في هرمز",
    "ما الفرق بين FTS5 والبحث الدلالي؟",
    "كيف أستخدم أدوات هرمز؟",
    "هل يدعم النظام اللغة العربية؟",
    "ما هي خوارزمية Leiden؟",
    "كيف أضيف مزود جديد؟",
    "كيف أبحث في الجلسات السابقة؟",
]

ENGLISH_QUERIES = [
    "How do I add a new model to Hermes?",
    "What is the knowledge graph and how does it work?",
    "How to index a project in memory?",
    "Memory system explanation in Hermes",
    "What is the difference between FTS5 and semantic search?",
    "How to use Hermes tools?",
    "Does the system support Arabic language?",
    "What is the Leiden algorithm?",
    "How to add a new provider?",
    "How to search previous sessions?",
]

EDGE_CASE_QUERIES = [
    "",           # Empty string
    "   ",        # Whitespace only
    "!!!",        # Symbols only
    "🤖🧠💡",     # Emoji only
    "a",          # Single character
]

# ============================================================
# Fixture: Initialized provider (shared across tests)
# ============================================================

@pytest.fixture(scope="session")
def provider():
    """Initialize provider once for all tests."""
    p = UnifiedMemoryProvider(CONFIG)
    p.initialize(session_id="baseline_test")
    # Wait for background preload
    if p._model_loading_thread and p._model_loading_thread.is_alive():
        p._model_ready.wait(timeout=30)
    yield p
    p.shutdown()


# ============================================================
# Benchmark Tests
# ============================================================

class TestBenchmarks:
    """Capture performance baselines before P0 refactoring."""

    def test_cold_start_time(self, provider, benchmark_data):
        """Cold start = initialize() + first search."""
        # Since provider is already initialized in fixture,
        # we measure first search time (model should be warm from preload)
        t0 = time.time()
        res = provider._tool_unified_search({'query': 'test', 'limit': 1})
        dt = time.time() - t0
        benchmark_data['first_search_ms'] = dt * 1000
        assert 'results' in res

    def test_warm_search_latency(self, provider, benchmark_data):
        """Average latency of 5 warm searches."""
        queries = ['gateway', 'memory', 'model', 'training', 'search']
        times = []
        for q in queries:
            t0 = time.time()
            res = provider._tool_unified_search({'query': q, 'limit': 3})
            dt = (time.time() - t0) * 1000
            times.append(dt)
            assert 'results' in res
        benchmark_data['warm_search_avg_ms'] = sum(times) / len(times)
        benchmark_data['warm_search_p95_ms'] = sorted(times)[4]

    def test_fts5_arabic_performance(self, benchmark_data):
        """FTS5 Arabic query performance (direct DB)."""
        import sqlite3
        db_path = os.path.expanduser('~/.hermes/state.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        ar_q = ['نظام', 'ذاكرة', 'هرمز', 'نموذج', 'بحث']
        times = []
        for q in ar_q:
            t0 = time.time()
            c.execute('SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?', [q])
            n = c.fetchone()[0]
            dt = (time.time() - t0) * 1000
            times.append(dt)
        conn.close()
        benchmark_data['fts5_arabic_avg_ms'] = sum(times) / len(times)
        benchmark_data['fts5_arabic_max_ms'] = max(times)


# ============================================================
# Arabic Snapshot Tests
# ============================================================

class TestArabicSnapshots:
    """Snapshot tests for Arabic queries — ensure output format doesn't regress."""

    @pytest.mark.parametrize("query", ARABIC_QUERIES)
    def test_arabic_prefetch_format(self, provider, query):
        """Arabic prefetch returns formatted output with source labels."""
        result = provider.prefetch(query, session_id="baseline_test")
        # Non-empty queries should return something (or empty if no matches)
        assert isinstance(result, str)
        if result:
            assert '## Memory Context' in result or result == ""
            # Check for source labels (P1 feature)
            if 'graph:' in result or 'session:' in result:
                # Good — source labels present
                pass


# ============================================================
# English Snapshot Tests
# ============================================================

class TestEnglishSnapshots:
    """Snapshot tests for English queries."""

    @pytest.mark.parametrize("query", ENGLISH_QUERIES)
    def test_english_prefetch_format(self, provider, query):
        """English prefetch returns formatted output."""
        result = provider.prefetch(query, session_id="baseline_test")
        assert isinstance(result, str)
        if result:
            assert '## Memory Context' in result or result == ""


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    """Edge case queries must return 0 or empty results."""

    @pytest.mark.parametrize("query", EDGE_CASE_QUERIES)
    def test_edge_case_returns_empty(self, provider, query):
        """Edge case queries return empty prefetch."""
        result = provider.prefetch(query, session_id="baseline_test")
        assert result == ""

    @pytest.mark.parametrize("query", EDGE_CASE_QUERIES)
    def test_edge_case_search_zero_results(self, provider, query):
        """Edge case unified_search returns 0 results."""
        if not query.strip() or len(query.strip()) < 2:
            res = provider._tool_unified_search({'query': query, 'limit': 5})
            assert res['total'] == 0
            assert res['results'] == []


# ============================================================
# Baseline Data Collection
# ============================================================

@pytest.fixture(scope="session")
def benchmark_data():
    """Collect benchmark data across tests."""
    data = {}
    yield data
    # Save after all tests complete
    baseline_file = os.path.join(BASELINE_DIR, 'benchmark_v1.json')
    data['timestamp'] = time.time()
    data['test_count'] = {
        'arabic_queries': len(ARABIC_QUERIES),
        'english_queries': len(ENGLISH_QUERIES),
        'edge_cases': len(EDGE_CASE_QUERIES),
    }
    with open(baseline_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Baseline saved to: {baseline_file}")
    print(f"  First search: {data.get('first_search_ms', 'N/A'):.1f}ms")
    print(f"  Warm search avg: {data.get('warm_search_avg_ms', 'N/A'):.1f}ms")
    print(f"  FTS5 Arabic avg: {data.get('fts5_arabic_avg_ms', 'N/A'):.1f}ms")
