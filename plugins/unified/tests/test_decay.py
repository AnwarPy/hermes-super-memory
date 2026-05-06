"""P1a: Decay System Tests

Tests for time-based decay scoring:
- Decay formula: adjusted = base * e^(-λ * age_days)
- λ = 0.01/day → half-life ≈ 69 days
- Results without timestamp get decay_factor = 1.0 (no decay)
"""

import math
import time
import pytest

from unified import UnifiedMemoryProvider


class TestDecayFormula:
    """Unit tests for the decay formula itself."""

    def test_decay_zero_age(self):
        """Age 0 = no decay (factor = 1.0)."""
        lambda_decay = 0.01
        age_days = 0
        factor = math.exp(-lambda_decay * age_days)
        assert abs(factor - 1.0) < 1e-10

    def test_decay_half_life(self):
        """At half-life (69.3 days), factor ≈ 0.5."""
        lambda_decay = 0.01
        age_days = 69.3  # ln(2) / 0.01 ≈ 69.3
        factor = math.exp(-lambda_decay * age_days)
        assert 0.49 < factor < 0.51

    def test_decay_30_days(self):
        """After 30 days with λ=0.01, factor ≈ 0.74."""
        lambda_decay = 0.01
        age_days = 30
        factor = math.exp(-lambda_decay * age_days)
        assert 0.73 < factor < 0.75

    def test_decay_180_days(self):
        """After 180 days with λ=0.01, factor ≈ 0.165."""
        lambda_decay = 0.01
        age_days = 180
        factor = math.exp(-lambda_decay * age_days)
        assert 0.15 < factor < 0.17

    def test_decay_zero_lambda(self):
        """λ=0 means no decay ever (factor always 1.0)."""
        lambda_decay = 0.0
        for age in [0, 30, 365, 10000]:
            factor = math.exp(-lambda_decay * age)
            assert abs(factor - 1.0) < 1e-10


class TestRrfDecayIntegration:
    """Integration tests for decay inside _rerank_rrf."""

    def test_no_decay_when_lambda_zero(self):
        """When lambda_decay=0, all results should have equal treatment."""
        config = {'graphs_dir': '/tmp', 'decay_lambda': 0.0}
        provider = UnifiedMemoryProvider(config)

        now = time.time()
        graph_results = [{'content': 'graph fact', 'similarity': 0.9}]
        fts5_results = [{
            'content': 'session message',
            'similarity': 0.75,
            'timestamp': now,  # Fresh
        }]

        ranked = provider._rerank_rrf(
            graph_results, fts5_results,
            k=60, graph_weight=1.5, fts5_weight=1.0, lambda_decay=0.0
        )
        assert len(ranked) == 2
        # Both should have adjusted_score
        for r in ranked:
            assert 'adjusted_score' in r

    def test_decay_reduces_old_fts5_score(self):
        """Old FTS5 results should score lower than fresh ones."""
        config = {'graphs_dir': '/tmp', 'decay_lambda': 0.01}
        provider = UnifiedMemoryProvider(config)

        now = time.time()
        old_ts = now - (180 * 86400)  # 180 days ago

        graph_results = [{'content': 'graph fact', 'similarity': 0.9}]
        fts5_fresh = [{
            'content': 'fresh session',
            'similarity': 0.75,
            'timestamp': now,
        }]
        fts5_old = [{
            'content': 'old session',
            'similarity': 0.75,
            'timestamp': old_ts,
        }]

        # Fresh should score higher than old (same similarity, different age)
        ranked_fresh = provider._rerank_rrf(
            graph_results, fts5_fresh, k=60, lambda_decay=0.01
        )
        ranked_old = provider._rerank_rrf(
            graph_results, fts5_old, k=60, lambda_decay=0.01
        )

        fresh_fts5_score = ranked_fresh[1]['rrf_score'] if len(ranked_fresh) > 1 else 0
        old_fts5_score = ranked_old[1]['rrf_score'] if len(ranked_old) > 1 else 0
        assert fresh_fts5_score > old_fts5_score, \
            f"Fresh score {fresh_fts5_score} should > old score {old_fts5_score}"

    def test_no_timestamp_means_no_decay(self):
        """Results without timestamp should not be decayed."""
        config = {'graphs_dir': '/tmp', 'decay_lambda': 0.01}
        provider = UnifiedMemoryProvider(config)

        graph_results = [{'content': 'no timestamp fact', 'similarity': 0.9}]
        fts5_results = []

        ranked = provider._rerank_rrf(
            graph_results, fts5_results, k=60, lambda_decay=0.01
        )
        assert len(ranked) == 1
        # Graph facts without timestamp should have full score
        r = ranked[0]
        assert r['rrf_score'] > 0
        assert r['adjusted_score'] == r['rrf_score']

    def test_decay_does_not_affect_graph_results(self):
        """Graph results (no timestamp) should not decay, only FTS5."""
        config = {'graphs_dir': '/tmp', 'decay_lambda': 0.01}
        provider = UnifiedMemoryProvider(config)

        graph_results = [
            {'content': 'graph fact A', 'similarity': 0.9},
            {'content': 'graph fact B', 'similarity': 0.8},
        ]
        fts5_results = []

        ranked_with_decay = provider._rerank_rrf(
            graph_results, fts5_results, k=60, lambda_decay=0.01
        )
        ranked_without_decay = provider._rerank_rrf(
            graph_results, fts5_results, k=60, lambda_decay=0.0
        )

        # Graph-only results should be identical regardless of decay
        assert len(ranked_with_decay) == len(ranked_without_decay)
        for r1, r2 in zip(ranked_with_decay, ranked_without_decay):
            assert r1['rrf_score'] == r2['rrf_score']
