"""P2: Utils Tests — full coverage for _format_age and _clean_chunk."""

import sys
import os

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.utils import _format_age, _clean_chunk


class TestFormatAge:
    """Test _format_age for all time ranges."""

    def test_now(self):
        assert _format_age(0) == "now"
        assert _format_age(30) == "now"
        assert _format_age(59) == "now"

    def test_minutes(self):
        assert _format_age(60) == "1m ago"
        assert _format_age(120) == "2m ago"
        assert _format_age(3599) == "59m ago"

    def test_hours(self):
        assert _format_age(3600) == "1h ago"
        assert _format_age(7200) == "2h ago"
        assert _format_age(86399) == "23h ago"

    def test_days(self):
        assert _format_age(86400) == "1d ago"
        assert _format_age(172800) == "2d ago"
        assert _format_age(604800) == "7d ago"

    def test_large_values(self):
        assert _format_age(31536000) == "365d ago"  # 1 year


class TestCleanChunk:
    """Test _clean_chunk for all edge cases."""

    def test_empty_input(self):
        assert _clean_chunk("") == ""
        assert _clean_chunk(None) == ""

    def test_removes_fts5_markers(self):
        result = _clean_chunk(">>>hello world<<< test data here")
        assert ">>>" not in result
        assert "<<<" not in result
        assert "hello" in result

    def test_removes_control_chars(self):
        result = _clean_chunk("hello\x00\x01\x02world test data here now")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "hello" in result
        assert "world" in result

    def test_normalizes_whitespace(self):
        result = _clean_chunk("hello   world   test data here")
        assert "  " not in result
        assert result == "hello world test data here"

    def test_handles_escaped_newlines(self):
        result = _clean_chunk("hello\\nworld\\r\\nmore")
        assert "\\n" not in result
        assert "\\r" not in result
        assert "hello" in result
        assert "world" in result

    def test_truncates_long_text(self):
        long_text = "a" * 500
        result = _clean_chunk(long_text, max_len=100)
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")

    def test_truncates_at_word_boundary(self):
        text = "word1 word2 word3 word4 word5" * 20
        result = _clean_chunk(text, max_len=50)
        # Should end with "..." and truncate at word boundary
        assert result.endswith("...")
        # Should not cut mid-word
        assert "word" in result

    def test_returns_empty_for_too_short(self):
        assert _clean_chunk("hi") == ""
        assert _clean_chunk("a" * 14) == ""

    def test_preserves_minimum_length(self):
        assert _clean_chunk("a" * 15) != ""

    def test_handles_tabs(self):
        result = _clean_chunk("hello\\tworld")
        assert "\\t" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = _clean_chunk("  hello world this is a test  ")
        assert result == "hello world this is a test"

    def test_combined_cleaning(self):
        text = "  >>>hello\x00  world<<<\\n  more\\tdata  "
        result = _clean_chunk(text)
        assert ">>>" not in result
        assert "<<<" not in result
        assert "\x00" not in result
        assert result == "hello world more data"
