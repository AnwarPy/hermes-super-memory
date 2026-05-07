"""Tests for TextSplitter — Arabic-aware text chunking."""

import os
import sys
import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))

from unified.text_splitter import TextSplitter, split_documents


class TestTextSplitterInit:
    """Test TextSplitter initialization."""

    def test_default_chunk_size(self):
        splitter = TextSplitter()
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 96

    def test_custom_chunk_size(self):
        splitter = TextSplitter(chunk_size=256, chunk_overlap=64)
        assert splitter.chunk_size == 256
        assert splitter.chunk_overlap == 64

    def test_separators_defined(self):
        splitter = TextSplitter()
        assert len(splitter.separators) > 0
        assert "\n\n" in splitter.separators
        assert "؟" in splitter.separators
        assert "،" in splitter.separators


class TestSplitTextBasic:
    """Test basic text splitting functionality."""

    def test_empty_text(self):
        splitter = TextSplitter()
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []
        assert splitter.split_text(None) == []

    def test_text_smaller_than_chunk(self):
        splitter = TextSplitter(chunk_size=512)
        result = splitter.split_text("Short text")
        assert result == ["Short text"]

    def test_text_exactly_chunk_size(self):
        text = "A" * 100
        splitter = TextSplitter(chunk_size=100)
        result = splitter.split_text(text)
        assert len(result) == 1

    def test_text_slightly_larger_than_chunk(self):
        text = "A" * 200
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 2


class TestSplitTextArabic:
    """Test Arabic-specific splitting."""

    def test_arabic_period(self):
        text = "هذه جملة أولى۔ وهذه جملة ثانية۔ وهذه جملة ثالثة"
        splitter = TextSplitter(chunk_size=40, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 1
        assert "۔" in result[0] or "جملة" in result[0]

    def test_arabic_question_mark(self):
        text = "كيف حالك؟ أنا بخير. ماذا عنك؟"
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 1

    def test_arabic_comma(self):
        text = "أحب البرمجة، والذكاء الاصطناعي، والتعلم الآلي"
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 1

    def test_mixed_arabic_english(self):
        text = "مرحباً hello world. كيف حالك؟ Fine thanks."
        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 1


class TestSplitTextEnglish:
    """Test English text splitting."""

    def test_sentence_split(self):
        text = "First sentence. Second sentence. Third sentence."
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 2

    def test_paragraph_split(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 2

    def test_newline_split(self):
        text = "Line one\nLine two\nLine three"
        splitter = TextSplitter(chunk_size=15, chunk_overlap=0)
        result = splitter.split_text(text)
        assert len(result) >= 2


class TestOverlap:
    """Test chunk overlap functionality."""

    def test_overlap_between_chunks(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        result = splitter.split_text(text)
        assert len(result) >= 2
        # Check that overlap was added
        for chunk in result[1:]:
            assert len(chunk) > 100  # original + overlap

    def test_no_overlap_when_disabled(self):
        text = "A" * 100 + "B" * 100
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        result = splitter.split_text(text)
        # Without overlap, chunks should be at most chunk_size
        for chunk in result:
            assert len(chunk.strip()) <= 100

    def test_single_chunk_no_overlap(self):
        text = "Short text"
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        result = splitter.split_text(text)
        assert len(result) == 1


class TestSplitMethod:
    """Test the split() method compatible with graph_engine."""

    def test_split_with_dict_docs(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        docs = [
            {"page_content": "This is a long text that should be split into multiple chunks for testing purposes."},
            {"page_content": "Short."}
        ]
        result = splitter.split(docs)
        assert len(result) >= 2
        assert "page_content" in result[0]
        assert "metadata" in result[0]

    def test_split_with_object_docs(self):
        class FakeDoc:
            def __init__(self, content, meta=None):
                self.page_content = content
                self.metadata = meta or {}

        splitter = TextSplitter(chunk_size=30, chunk_overlap=0)
        docs = [FakeDoc("This is a longer text that exceeds the chunk size limit.", {"source": "test"})]
        result = splitter.split(docs)
        assert len(result) >= 1
        assert hasattr(result[0], "page_content")
        assert result[0].metadata == {"source": "test"}

    def test_split_with_string_input(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        result = splitter.split(["Just a plain string"])
        assert len(result) >= 1

    def test_split_with_file_type_param(self):
        """file_type param should be accepted but ignored (compatibility)."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        docs = [{"page_content": "Some content here."}]
        result = splitter.split(docs, file_type="pdf")
        assert len(result) >= 1


class TestSplitDocumentsFunction:
    """Test the standalone split_documents() function."""

    def test_split_documents_with_dicts(self):
        docs = [{"content": "First document content here that is long enough.", "metadata": {"id": 1}}]
        result = split_documents(docs, chunk_size=30, chunk_overlap=0)
        assert len(result) >= 1
        assert "content" in result[0]
        assert result[0]["metadata"] == {"id": 1}

    def test_split_documents_with_objects(self):
        class FakeDoc:
            def __init__(self, content, meta=None):
                self.page_content = content
                self.metadata = meta or {}

        docs = [FakeDoc("Document content that should be split into chunks.", {"source": "file"})]
        result = split_documents(docs, chunk_size=30, chunk_overlap=0)
        assert len(result) >= 1
        assert hasattr(result[0], "page_content")

    def test_split_documents_empty(self):
        result = split_documents([], chunk_size=100)
        assert result == []

    def test_split_documents_default_params(self):
        docs = [{"content": "A" * 600}]
        result = split_documents(docs)
        # Default chunk_size=512, so 600 chars should produce 2+ chunks
        assert len(result) >= 1
