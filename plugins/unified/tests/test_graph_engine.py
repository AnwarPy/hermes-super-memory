"""Tests for GraphifyEngine — core knowledge graph engine."""

import os
import sys
import json
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))


@pytest.fixture
def tmp_graphs_dir():
    """Create a temporary directory for graph storage."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def clear_embedding_singleton():
    """Clear the model singleton cache before each test."""
    try:
        from unified.embedding_model import _MODEL_SINGLETON
        _MODEL_SINGLETON.clear()
    except ImportError:
        pass
    yield
    try:
        from unified.embedding_model import _MODEL_SINGLETON
        _MODEL_SINGLETON.clear()
    except ImportError:
        pass


class TestGraphifyEngineInit:
    """Test engine initialization."""

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_default_config(self, mock_builder, mock_loader, mock_embedding):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine()
        assert engine.chunk_size == 512
        assert engine.chunk_overlap == 96
        assert engine.similarity_threshold == 0.7
        assert engine.community_algorithm == "leiden"

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_custom_config(self, mock_builder, mock_loader, mock_embedding):
        from unified.graph_engine import GraphifyEngine
        config = {
            "chunk_size": 256,
            "chunk_overlap": 32,
            "similarity_threshold": 0.8,
            "community_algorithm": "louvain",
        }
        engine = GraphifyEngine(config=config)
        assert engine.chunk_size == 256
        assert engine.chunk_overlap == 32
        assert engine.similarity_threshold == 0.8
        assert engine.community_algorithm == "louvain"

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_components_initialized(self, mock_builder, mock_loader, mock_embedding):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine()
        assert engine.loader is not None
        assert engine.splitter is not None
        assert engine.embedding is not None
        assert engine.builder is not None
        assert engine.detector is not None
        assert engine.storage is not None


class TestGraphifyEngineConstants:
    """Test safety limit constants."""

    def test_max_nodes_hard(self):
        from unified.graph_engine import MAX_NODES_HARD
        assert MAX_NODES_HARD == 2500

    def test_max_nodes_warning(self):
        from unified.graph_engine import MAX_NODES_WARNING
        assert MAX_NODES_WARNING == 2000

    def test_max_edges_warning(self):
        from unified.graph_engine import MAX_EDGES_WARNING
        assert MAX_EDGES_WARNING == 10000


class TestSearchSemantic:
    """Test semantic search functionality."""

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_search_returns_empty_for_missing_project(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})
        result = engine.search_semantic("test query", "nonexistent_project")
        assert result == []

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_search_returns_results_with_graph(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        import networkx as nx
        import numpy as np
        from unified.graph_engine import GraphifyEngine

        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})

        # Mock embedding to return a fixed vector
        mock_embedding.return_value.embed_query.return_value = [0.1] * 1024

        # Create a test graph with embeddings
        graph = nx.Graph()
        graph.add_node("node_1", content="Test content", type="fact", embedding=[0.1] * 1024)
        graph.add_node("node_2", content="Another test", type="fact", embedding=[0.2] * 1024)

        # Save the graph
        engine.storage.save(graph, {"num_communities": 0, "modularity": 0}, "test_project")

        # Mock the storage load
        with patch.object(engine.storage, 'load', return_value=graph):
            results = engine.search_semantic("test", "test_project", top_k=5)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert "node_id" in results[0]
        assert "content" in results[0]
        assert "similarity" in results[0]


class TestGetGraphStats:
    """Test graph statistics retrieval."""

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_stats_for_missing_project(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})
        result = engine.get_graph_stats("nonexistent")
        assert "error" in result

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_stats_for_existing_graph(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        import networkx as nx
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})

        graph = nx.Graph()
        graph.add_node("n1", content="test", type="fact")
        graph.add_node("n2", content="test2", type="fact")
        graph.add_edge("n1", "n2", weight=0.8, type="similar")

        with patch.object(engine.storage, 'load', return_value=graph):
            with patch.object(engine.storage, 'load_communities', return_value={"num_communities": 1, "modularity": 0.5}):
                result = engine.get_graph_stats("test_project")

        assert result["num_nodes"] == 2
        assert result["num_edges"] == 1
        assert result["num_communities"] == 1
        assert result["modularity"] == 0.5


class TestListProjects:
    """Test project listing."""

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_list_projects_empty(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})
        result = engine.list_projects()
        assert isinstance(result, list)


class TestIndexDirectoryEdgeCases:
    """Test index_directory edge cases without loading actual model."""

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_raises_for_nonexistent_path(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})
        with pytest.raises(FileNotFoundError):
            engine.index_directory("/nonexistent/path/that/does/not/exist")

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_returns_no_documents_for_empty_dir(self, mock_builder, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})

        # Mock loader to return no documents
        mock_loader.return_value.load_directory.return_value = []

        result = engine.index_directory(tmp_graphs_dir, project_name="empty_test")
        assert result["status"] == "no_documents"
        assert result["project_name"] == "empty_test"

    @patch('unified.graph_engine.EmbeddingModel')
    @patch('unified.graph_engine.DocumentLoader')
    @patch('unified.graph_engine.KnowledgeGraphBuilder')
    def test_raises_for_too_many_nodes(self, mock_builder_cls, mock_loader, mock_embedding, tmp_graphs_dir):
        from unified.graph_engine import GraphifyEngine, MAX_NODES_HARD
        engine = GraphifyEngine(config={"graphs_dir": tmp_graphs_dir})

        # Mock loader to return one doc
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {"source": "test.txt"}
        mock_loader.return_value.load_directory.return_value = [mock_doc]

        # Mock splitter to return objects with page_content (matching langchain Document)
        class FakeChunk:
            def __init__(self):
                self.page_content = "chunk"
                self.metadata = {}

        with patch.object(engine.splitter, 'split', return_value=[FakeChunk()]):
            # The index_directory method recreates the builder internally, so we must
            # configure the mock class return value
            mock_builder_instance = MagicMock()
            mock_builder_instance.add_nodes_from_docs.return_value = ["node_%d" % i for i in range(MAX_NODES_HARD + 1)]
            mock_builder_instance.add_edges_by_similarity.return_value = 100  # Below warning threshold
            mock_builder_cls.return_value = mock_builder_instance

            with pytest.raises(ValueError) as exc_info:
                engine.index_directory(tmp_graphs_dir, project_name="big_test")

            assert "يتجاوز الحد الأقصى" in str(exc_info.value) or "exceeds" in str(exc_info.value).lower()
