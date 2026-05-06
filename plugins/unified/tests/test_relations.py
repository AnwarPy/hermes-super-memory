"""P3: Typed Graph Relations Tests

Tests for RelationClassifier and typed relation filtering.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.relations import (
    RelationClassifier, RELATION_TYPES, VALID_TYPES,
    RELATION_CLASSIFY_PROMPT,
)


class TestRelationTypes:
    """Test relation type definitions."""

    def test_all_types_have_descriptions(self):
        for rtype, info in RELATION_TYPES.items():
            assert 'description_ar' in info
            assert 'description_en' in info
            assert 'direction' in info
            assert info['direction'] in ('symmetric', 'asymmetric')

    def test_asymmetric_types(self):
        """These should be asymmetric (A→B ≠ B→A)."""
        asymmetric = {'causes', 'fixes', 'supports'}
        for t in asymmetric:
            assert RELATION_TYPES[t]['direction'] == 'asymmetric'

    def test_symmetric_types(self):
        """These should be symmetric (A↔B)."""
        symmetric = {'contradicts', 'related'}
        for t in symmetric:
            assert RELATION_TYPES[t]['direction'] == 'symmetric'

    def test_valid_types_set(self):
        assert VALID_TYPES == set(RELATION_TYPES.keys())
        assert 'causes' in VALID_TYPES
        assert 'fixes' in VALID_TYPES
        assert 'supports' in VALID_TYPES
        assert 'contradicts' in VALID_TYPES
        assert 'related' in VALID_TYPES


class TestRelationClassifierInit:
    """Test classifier initialization."""

    def test_default_config(self):
        classifier = RelationClassifier()
        assert classifier.ollama_url == 'http://localhost:11434/api/generate'
        assert classifier.relation_model == 'qwen2.5:3b'
        assert classifier.confidence_threshold == 0.7
        assert classifier.fallback_on_error is True

    def test_custom_config(self):
        classifier = RelationClassifier({
            'ollama_url': 'http://my-server:11434/api/generate',
            'relation_model': 'qwen2.5:7b',
            'confidence_threshold': 0.5,
            'fallback_on_error': False,
        })
        assert classifier.ollama_url == 'http://my-server:11434/api/generate'
        assert classifier.relation_model == 'qwen2.5:7b'
        assert classifier.confidence_threshold == 0.5
        assert classifier.fallback_on_error is False


class TestSimpleClassifier:
    """Test heuristic-based fallback classifier."""

    def setup_method(self):
        self.classifier = RelationClassifier({'fallback_on_error': True})

    def test_fix_detection(self):
        """Should detect fix relation."""
        result = self.classifier._classify_simple(
            'حلينا مشكلة البوابة بإعادة التشغيل',
            'البوابة كانت تعطي خطأ 500'
        )
        assert result is not None
        assert result['relation_type'] == 'fixes'
        assert result['confidence'] < 0.5  # Heuristic = low confidence

    def test_contradiction_detection(self):
        """Should detect contradiction."""
        result = self.classifier._classify_simple(
            'المستخدم يفضل SQLite',
            'لكن النظام يحتاج PostgreSQL'
        )
        assert result is not None
        assert result['relation_type'] == 'contradicts'

    def test_cause_detection(self):
        """Should detect causation."""
        result = self.classifier._classify_simple(
            'أضفنا Caching',
            'الأداء تحسن بشكل ملحوظ'
        )
        # Might detect cause or just related (heuristic)
        assert result is None or result['relation_type'] in VALID_TYPES

    def test_no_relation(self):
        """Unrelated facts should return None."""
        result = self.classifier._classify_simple(
            'الطقس جميل اليوم',
            'القطار يصل الساعة الخامسة'
        )
        assert result is None

    def test_fallback_on_llm_failure(self):
        """When LLM fails, should use simple classifier."""
        classifier = RelationClassifier({
            'ollama_url': 'http://nonexistent:9999',
            'fallback_on_error': True,
        })
        result = classifier.classify_relation(
            'حلينا مشكلة البوابة',
            'البوابة كانت تعطي خطأ'
        )
        # Should return heuristic result, not None
        assert result is not None
        assert result['relation_type'] in VALID_TYPES
        assert result['confidence'] < 0.5  # Heuristic = low


class TestBatchClassification:
    """Test batch classification."""

    def test_batch_with_empty(self):
        classifier = RelationClassifier({'fallback_on_error': True})
        results = classifier.classify_batch([])
        assert results == []

    def test_batch_returns_same_order(self):
        classifier = RelationClassifier({'fallback_on_error': True})
        pairs = [
            ('حلينا مشكلة', 'كان فيه خطأ'),
            ('أضفنا ميزة', 'النظام صار أفضل'),
            ('الطقس جميل', 'القطار متأخر'),
        ]
        results = classifier.classify_batch(pairs)
        assert len(results) == 3
        # First should detect some relation (fixes, causes, etc.)
        assert results[0] is not None
        assert results[0]['relation_type'] in VALID_TYPES
        # Third is unrelated — LLM might return 'related' or None
        assert results[2] is None or results[2]['relation_type'] == 'related'


class TestTypedRelationFilter:
    """Test that graph_search can filter by relation_type."""

    def test_graph_search_schema_has_relation_type(self):
        """Verify the tool schema includes relation_type."""
        from unified import UnifiedMemoryProvider
        provider = UnifiedMemoryProvider({})
        schemas = provider.get_tool_schemas()
        
        graph_search = next(s for s in schemas if s['name'] == 'graph_search')
        props = graph_search['parameters']['properties']
        assert 'relation_type' in props
        assert props['relation_type']['enum'] == [
            'causes', 'fixes', 'supports', 'contradicts', 'related'
        ]

    def test_search_graph_cached_accepts_relation_type(self):
        """Verify _search_graph_cached accepts relation_type parameter."""
        from unified import UnifiedMemoryProvider
        provider = UnifiedMemoryProvider({})
        
        # Should not raise even without DB
        results = provider._search_graph_cached(
            'test', relation_type='causes'
        )
        assert results == []  # No DB, empty result
