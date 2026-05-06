"""P5: Agent-ID Tests — full coverage for AgentIdMixin."""

import sys
import os

import pytest

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.agent_id import (
    AgentIdMixin, DEFAULT_AGENT_ID, AGENT_ID_KEY,
)


class MockProvider(AgentIdMixin):
    """Mock class that inherits from AgentIdMixin for testing."""
    pass


class TestAgentIdDefaults:
    """Test default values."""

    def test_default_agent_id(self):
        assert DEFAULT_AGENT_ID == "default"

    def test_agent_id_key(self):
        assert AGENT_ID_KEY == "agent_id"


class TestAgentIdMixinInit:
    """Test initialization."""

    def test_init_sets_default(self):
        provider = MockProvider()
        assert provider._agent_id == DEFAULT_AGENT_ID

    def test_get_agent_id(self):
        provider = MockProvider()
        assert provider._get_agent_id() == DEFAULT_AGENT_ID


class TestSetAgentId:
    """Test setting agent ID."""

    def test_set_valid_id(self):
        provider = MockProvider()
        provider._set_agent_id("agent-5")
        assert provider._agent_id == "agent-5"

    def test_set_with_whitespace(self):
        provider = MockProvider()
        provider._set_agent_id("  agent-3  ")
        assert provider._agent_id == "agent-3"

    def test_set_empty_id_resets(self):
        provider = MockProvider()
        provider._set_agent_id("agent-5")
        provider._set_agent_id("")
        assert provider._agent_id == DEFAULT_AGENT_ID

    def test_set_none_resets(self):
        provider = MockProvider()
        provider._set_agent_id("agent-5")
        provider._set_agent_id(None)
        assert provider._agent_id == DEFAULT_AGENT_ID

    def test_set_multiple_times(self):
        provider = MockProvider()
        provider._set_agent_id("agent-1")
        assert provider._agent_id == "agent-1"
        provider._set_agent_id("agent-2")
        assert provider._agent_id == "agent-2"


class TestAnnotateWithAgentId:
    """Test adding agent_id to results."""

    def test_annotate_single_result(self):
        provider = MockProvider()
        provider._set_agent_id("agent-5")

        result = {"content": "test", "similarity": 0.9}
        annotated = provider._annotate_with_agent_id(result)

        assert annotated["agent_id"] == "agent-5"
        assert annotated["content"] == "test"

    def test_annotate_none(self):
        provider = MockProvider()
        result = provider._annotate_with_agent_id(None)
        assert result is None

    def test_annotate_non_dict(self):
        provider = MockProvider()
        result = provider._annotate_with_agent_id("not a dict")
        assert result == "not a dict"

    def test_annotate_empty_dict(self):
        provider = MockProvider()
        provider._set_agent_id("agent-1")
        result = provider._annotate_with_agent_id({})
        assert result["agent_id"] == "agent-1"


class TestAnnotateResultsWithAgentId:
    """Test adding agent_id to result lists."""

    def test_annotate_list(self):
        provider = MockProvider()
        provider._set_agent_id("agent-7")

        results = [
            {"content": "a"},
            {"content": "b"},
            {"content": "c"},
        ]
        annotated = provider._annotate_results_with_agent_id(results)

        assert len(annotated) == 3
        assert all(r["agent_id"] == "agent-7" for r in annotated)

    def test_annotate_empty_list(self):
        provider = MockProvider()
        results = provider._annotate_results_with_agent_id([])
        assert results == []


class TestFilterByAgentId:
    """Test filtering results by agent_id."""

    def test_filter_none_returns_all(self):
        provider = MockProvider()
        results = [
            {"agent_id": "agent-1", "content": "a"},
            {"agent_id": "agent-2", "content": "b"},
        ]
        filtered = provider._filter_by_agent_id(results, agent_id=None)
        assert len(filtered) == 2

    def test_filter_all_returns_all(self):
        provider = MockProvider()
        results = [
            {"agent_id": "agent-1", "content": "a"},
            {"agent_id": "agent-2", "content": "b"},
        ]
        filtered = provider._filter_by_agent_id(results, agent_id="all")
        assert len(filtered) == 2

    def test_filter_star_returns_all(self):
        provider = MockProvider()
        results = [
            {"agent_id": "agent-1", "content": "a"},
        ]
        filtered = provider._filter_by_agent_id(results, agent_id="*")
        assert len(filtered) == 1

    def test_filter_matching(self):
        provider = MockProvider()
        results = [
            {"agent_id": "agent-1", "content": "a"},
            {"agent_id": "agent-2", "content": "b"},
            {"agent_id": "agent-1", "content": "c"},
        ]
        filtered = provider._filter_by_agent_id(results, agent_id="agent-1")
        assert len(filtered) == 2
        assert all(r["agent_id"] == "agent-1" for r in filtered)

    def test_filter_no_match(self):
        provider = MockProvider()
        results = [
            {"agent_id": "agent-1", "content": "a"},
            {"agent_id": "agent-2", "content": "b"},
        ]
        filtered = provider._filter_by_agent_id(results, agent_id="agent-99")
        assert filtered == []

    def test_filter_empty_list(self):
        provider = MockProvider()
        filtered = provider._filter_by_agent_id([], agent_id="agent-1")
        assert filtered == []


class TestGetAgentIdFromArgs:
    """Test extracting agent_id from arguments."""

    def test_get_from_args(self):
        provider = MockProvider()
        args = {"query": "test", "agent_id": "agent-5"}
        result = provider._get_agent_id_from_args(args)
        assert result == "agent-5"

    def test_get_missing_from_args(self):
        provider = MockProvider()
        args = {"query": "test"}
        result = provider._get_agent_id_from_args(args)
        assert result is None

    def test_get_empty_from_args(self):
        provider = MockProvider()
        args = {"query": "test", "agent_id": ""}
        result = provider._get_agent_id_from_args(args)
        assert result == ""


class TestGetAgentIdSchema:
    """Test agent_id schema generation."""

    def test_schema_structure(self):
        provider = MockProvider()
        schema = provider.get_agent_id_schema()

        assert "agent_id" in schema
        assert schema["agent_id"]["type"] == "string"
        assert "description" in schema["agent_id"]

    def test_schema_description_mentions_all(self):
        provider = MockProvider()
        schema = provider.get_agent_id_schema()
        desc = schema["agent_id"]["description"]
        assert "all" in desc.lower()


class TestGetAgentStats:
    """Test agent statistics."""

    def test_stats_has_agent_id(self):
        provider = MockProvider()
        provider._set_agent_id("agent-3")
        stats = provider.get_agent_stats()
        assert stats["agent_id"] == "agent-3"

    def test_stats_has_status(self):
        provider = MockProvider()
        stats = provider.get_agent_stats()
        assert stats["status"] == "active"

    def test_stats_default_agent(self):
        provider = MockProvider()
        stats = provider.get_agent_stats()
        assert stats["agent_id"] == DEFAULT_AGENT_ID
