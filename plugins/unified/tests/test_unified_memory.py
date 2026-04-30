"""Tests for __init__.py — Unified Memory Provider"""

import pytest
import numpy as np
from unified import _clean_chunk, QueryResultCache


# ============================================================
# _clean_chunk tests — Arabic + English + Edge cases
# ============================================================

class TestCleanChunk:
    """اختبار تنظيف المحتوى"""

    # --- Valid content that should be kept ---

    def test_keeps_valid_arabic(self):
        text = 'هذا نص عربي طويل جداً للتجربة والاختبار.'
        result = _clean_chunk(text)
        assert len(result) >= 25

    def test_keeps_valid_english(self):
        text = 'This is a valid English text with more than twenty five characters.'
        result = _clean_chunk(text)
        assert len(result) >= 25

    def test_keeps_mixed_content(self):
        text = 'نص عربي mixed with English for testing purposes.'
        result = _clean_chunk(text)
        assert len(result) >= 25

    def test_keeps_arabic_with_punctuation(self):
        text = 'نظام الذاكرة الموحد، يعمل بشكل صحيح.'
        result = _clean_chunk(text)
        assert len(result) >= 25

    # --- Content that should be filtered out ---

    def test_empty_returns_empty(self):
        assert _clean_chunk('') == ''

    def test_short_returns_empty(self):
        assert _clean_chunk('قصير') == ''

    def test_very_short_english(self):
        assert _clean_chunk('short') == ''

    def test_code_def_filtered(self):
        assert _clean_chunk('def hello(): pass') == ''

    def test_code_class_filtered(self):
        assert _clean_chunk('class Foo:') == ''

    def test_code_import_filtered(self):
        assert _clean_chunk('import os') == ''

    def test_code_from_filtered(self):
        assert _clean_chunk('from pathlib import Path') == ''

    def test_code_print_filtered(self):
        assert _clean_chunk('print("hello")') == ''

    def test_code_return_filtered(self):
        assert _clean_chunk('return result') == ''

    def test_code_self_filtered(self):
        assert _clean_chunk('self.method()') == ''

    def test_code_for_filtered(self):
        assert _clean_chunk('for x in items:') == ''

    def test_code_if_filtered(self):
        assert _clean_chunk('if condition:') == ''

    def test_code_try_filtered(self):
        assert _clean_chunk('try:') == ''

    def test_code_except_filtered(self):
        assert _clean_chunk('except Exception:') == ''

    def test_code_with_filtered(self):
        assert _clean_chunk('with open(file):') == ''

    def test_code_comment_filtered(self):
        assert _clean_chunk('# This is a comment') == ''

    def test_code_backtick_filtered(self):
        assert _clean_chunk('```python') == ''

    # --- Edge cases ---

    def test_whitespace_only(self):
        assert _clean_chunk('   ') == ''

    def test_leading_dots_filtered(self):
        text = '... نص عربي بعد نقاط.'
        result = _clean_chunk(text)
        # Leading dots should be stripped
        assert '...' != result[:3]

    def test_truncated_english_word_filtered(self):
        """كلمة إنجليزية قصيرة جداً في النهاية"""
        result = _clean_chunk('Graph S')
        assert result == ''

    def test_file_path_at_start(self):
        text = '/home/user/file.py some arabic text here for testing.'
        result = _clean_chunk(text)
        assert len(result) > 0  # Should find real content after path

    def test_max_len_truncation(self):
        text = 'هذا نص طويل جداً ' * 50  # 1000+ chars
        result = _clean_chunk(text, max_len=100)
        assert len(result) <= 120  # max_len + '...'


# ============================================================
# QueryResultCache tests
# ============================================================

class TestQueryResultCache:
    """اختبار كاش نتائج البحث"""

    def test_cache_miss(self):
        cache = QueryResultCache(ttl_seconds=60)
        assert cache.get('nonexistent') is None

    def test_cache_hit(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set('test', [{'content': 'result'}])
        result = cache.get('test')
        assert result is not None
        assert result[0]['content'] == 'result'

    def test_cache_ttl_expiry(self):
        import time
        cache = QueryResultCache(ttl_seconds=0)  # TTL = 0 = immediate expiry
        cache.set('test', [{'content': 'result'}])
        time.sleep(0.01)
        assert cache.get('test') is None

    def test_cache_clear(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set('a', [1])
        cache.set('b', [2])
        cache.clear()
        assert cache.get('a') is None
        assert cache.get('b') is None

    def test_cache_different_queries(self):
        cache = QueryResultCache(ttl_seconds=60)
        cache.set('query1', [{'content': 'a'}])
        cache.set('query2', [{'content': 'b'}])
        assert cache.get('query1')[0]['content'] == 'a'
        assert cache.get('query2')[0]['content'] == 'b'


# ============================================================
# Arabic normalizer tests
# ============================================================

class TestArabicNormalizer:
    """اختبار تطبيع النص العربي"""

    def test_normalize(self):
        from unified.arabic_normalizer import normalize_query
        result = normalize_query('الذاكرة')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_empty(self):
        from unified.arabic_normalizer import normalize_query
        result = normalize_query('')
        assert isinstance(result, str)
