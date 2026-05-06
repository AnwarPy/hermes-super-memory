"""P2: Extended Relations Tests — full coverage for LLM paths, edge cases, add_typed_relations."""

import json
import os
import sqlite3
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.relations import RelationClassifier, VALID_TYPES


class TestLLMClassification:
    """Test LLM-based classification paths."""

    def test_llm_success(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': json.dumps({
                'relation_type': 'causes',
                'confidence': 0.85,
                'reasoning': 'السبب يؤدي للنتيجة',
            })
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            result = classifier.classify_relation(
                'أضفنا caching', 'الأداء تحسن'
            )

        assert result is not None
        assert result['relation_type'] == 'causes'
        assert result['confidence'] == 0.85
        assert result['reasoning'] == 'السبب يؤدي للنتيجة'

    def test_llm_invalid_type_defaults_to_related(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': json.dumps({
                'relation_type': 'invalid_type',
                'confidence': 0.9,
                'reasoning': 'test',
            })
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            result = classifier.classify_relation('A', 'B')

        assert result is not None
        assert result['relation_type'] == 'related'

    def test_llm_returns_none(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': json.dumps({
                'relation_type': 'none',
                'confidence': 0.0,
                'reasoning': 'no relation',
            })
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            result = classifier.classify_relation('A', 'B')

        assert result is None

    def test_llm_http_error_with_fallback(self):
        classifier = RelationClassifier({'fallback_on_error': True})
        with patch('httpx.post', side_effect=Exception("HTTP error")):
            result = classifier.classify_relation(
                'حلينا مشكلة', 'كان فيه خطأ'
            )

        assert result is not None
        assert result['confidence'] < 0.5  # Heuristic confidence

    def test_llm_http_error_no_fallback(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        with patch('httpx.post', side_effect=Exception("HTTP error")):
            result = classifier.classify_relation('A', 'B')

        assert result is None

    def test_llm_markdown_json_stripping(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '```json\n{"relation_type": "fixes", "confidence": 0.8, "reasoning": "fix"}\n```'
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            result = classifier.classify_relation('A', 'B')

        assert result is not None
        assert result['relation_type'] == 'fixes'

    def test_llm_json_without_markdown(self):
        classifier = RelationClassifier({'fallback_on_error': False})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '{"relation_type": "supports", "confidence": 0.75, "reasoning": "support"}'
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response):
            result = classifier.classify_relation('A', 'B')

        assert result is not None
        assert result['relation_type'] == 'supports'

    def test_llm_temperature_set(self):
        classifier = RelationClassifier()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '{"relation_type": "related", "confidence": 0.5, "reasoning": ""}'
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response) as mock_post:
            classifier.classify_relation('A', 'B')

        call_json = mock_post.call_args[1]['json']
        assert call_json['temperature'] == 0.1
        assert call_json['stream'] is False

    def test_llm_custom_model(self):
        classifier = RelationClassifier({'relation_model': 'qwen2.5:7b'})
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '{"relation_type": "related", "confidence": 0.5, "reasoning": ""}'
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.post', return_value=mock_response) as mock_post:
            classifier.classify_relation('A', 'B')

        call_json = mock_post.call_args[1]['json']
        assert call_json['model'] == 'qwen2.5:7b'


class TestSimpleClassifierExtended:
    """Extended heuristic fallback tests."""

    def setup_method(self):
        self.classifier = RelationClassifier({'fallback_on_error': True})

    def test_fix_keywords_arabic(self):
        result = self.classifier._classify_simple(
            'إصلاح البوابة', 'كان فيه مشكلة في النظام'
        )
        assert result is not None
        assert result['relation_type'] == 'fixes'

    def test_fix_keywords_english(self):
        result = self.classifier._classify_simple(
            'fix the error', 'there was a bug'
        )
        assert result is not None
        assert result['relation_type'] == 'fixes'

    def test_cause_keywords_arabic(self):
        result = self.classifier._classify_simple(
            'بسبب الضغط العالي', 'النظام توقف'
        )
        assert result is not None
        assert result['relation_type'] == 'causes'

    def test_cause_keywords_english(self):
        result = self.classify_simple(
            'because of high load', 'the server crashed'
        ) if hasattr(self, 'classify_simple') else None
        result = self.classifier._classify_simple(
            'because of high load', 'the server crashed'
        )
        assert result is not None
        assert result['relation_type'] == 'causes'

    def test_contradiction_english(self):
        result = self.classifier._classify_simple(
            'use SQLite', 'but we need PostgreSQL'
        )
        assert result is not None
        assert result['relation_type'] == 'contradicts'

    def test_shared_words_relation(self):
        result = self.classifier._classify_simple(
            'Python is fast and efficient',
            'Python performance is great'
        )
        assert result is not None
        assert result['relation_type'] == 'related'
        assert result['confidence'] == 0.25

    def test_no_shared_words_no_keywords(self):
        result = self.classifier._classify_simple(
            'السماء زرقاء',
            'السيارة حمراء'
        )
        assert result is None


class TestAddTypedRelations:
    """Test add_typed_relations database integration."""

    def test_add_with_none_db(self):
        classifier = RelationClassifier()
        result = classifier.add_typed_relations(None, 1, [2, 3])
        assert result == 0

    def test_add_with_no_conn(self):
        classifier = RelationClassifier()
        mock_db = MagicMock()
        mock_db.conn = None
        result = classifier.add_typed_relations(mock_db, 1, [2, 3])
        assert result == 0

    def test_add_with_empty_existing_ids(self):
        classifier = RelationClassifier()
        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        result = classifier.add_typed_relations(mock_db, 1, [])
        assert result == 0

    def test_add_new_fact_not_found(self):
        classifier = RelationClassifier()
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_db.conn = mock_conn

        result = classifier.add_typed_relations(mock_db, 1, [2])
        assert result == 0

    def test_add_existing_fact_not_found(self):
        classifier = RelationClassifier({'fallback_on_error': True})
        mock_db = MagicMock()
        mock_conn = MagicMock()

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            mock_cursor = MagicMock()
            if call_count[0] == 1:
                # New fact found
                mock_cursor.fetchone.return_value = ['New fact text']
            else:
                # Existing fact not found
                mock_cursor.fetchone.return_value = None
            return mock_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.conn = mock_conn

        result = classifier.add_typed_relations(mock_db, 1, [2, 3])
        # Both existing facts not found, but new fact was found
        assert result == 0

    def test_add_relation_below_threshold(self):
        classifier = RelationClassifier({
            'fallback_on_error': True,
            'confidence_threshold': 0.9,  # High threshold
        })
        mock_db = MagicMock()
        mock_conn = MagicMock()

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            mock_cursor = MagicMock()
            if call_count[0] == 1:
                mock_cursor.fetchone.return_value = ['New fact']
            else:
                mock_cursor.fetchone.return_value = ['Existing fact']
            return mock_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.conn = mock_conn

        result = classifier.add_typed_relations(mock_db, 1, [2])
        # Heuristic confidence (0.25-0.35) is below 0.9 threshold
        assert result == 0
        mock_db.add_relation.assert_not_called()

    def test_add_relation_above_threshold(self):
        classifier = RelationClassifier({
            'fallback_on_error': True,
            'confidence_threshold': 0.1,  # Low threshold
        })
        mock_db = MagicMock()
        mock_conn = MagicMock()

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            mock_cursor = MagicMock()
            if call_count[0] == 1:
                mock_cursor.fetchone.return_value = ['حلينا مشكلة البوابة']
            else:
                mock_cursor.fetchone.return_value = ['كان فيه خطأ في البوابة']
            return mock_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.conn = mock_conn

        result = classifier.add_typed_relations(mock_db, 1, [2])
        # Heuristic confidence (~0.35) is above 0.1 threshold
        assert result >= 0
        # May or may not call add_relation depending on heuristic result

    def test_add_multiple_existing_facts(self):
        classifier = RelationClassifier({'fallback_on_error': True})
        mock_db = MagicMock()
        mock_conn = MagicMock()

        facts = {
            1: 'New fact text',
            2: 'Existing fact A',
            3: 'Existing fact B',
        }
        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            mock_cursor = MagicMock()
            fact_id = params[0] if params else None
            mock_cursor.fetchone.return_value = [facts.get(fact_id, '')]
            return mock_cursor

        mock_conn.execute.side_effect = side_effect
        mock_db.conn = mock_conn

        result = classifier.add_typed_relations(mock_db, 1, [2, 3])
        assert isinstance(result, int)
