"""Tests for graph_storage.py — حفظ وتحميل الرسوم"""

import pytest
import os
import json
import shutil
import networkx as nx
import numpy as np
from pathlib import Path
from unified.graph_storage import GraphStorage


@pytest.fixture
def tmp_dir(tmp_path):
    """مؤقت مجلد"""
    return str(tmp_path / 'graphs')


@pytest.fixture
def storage(tmp_dir):
    return GraphStorage(tmp_dir)


@pytest.fixture
def sample_graph():
    g = nx.Graph()
    for i in range(10):
        g.add_node(
            f'n{i}',
            content=f'Content {i}',
            embedding=np.random.randn(64).tolist(),
            metadata={'source': f'file{i}.txt'},
            type='text',
            created_at='2026-01-01T00:00:00',
        )
    for i in range(8):
        g.add_edge(f'n{i}', f'n{i+2}', weight=0.7, type='semantic')
    return g


# ============================================================
# save() tests
# ============================================================

class TestSave:
    def test_save_creates_graph_json(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        assert result['graph_json'].exists()

    def test_save_creates_communities_json(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        assert result['communities_json'].exists()

    def test_save_creates_metadata_json(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        assert result['metadata_json'].exists()

    def test_save_creates_report_when_enabled(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=True)
        assert result['report_md'].exists()

    def test_save_no_report_when_disabled(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        assert result['report_md'] is None

    def test_save_is_compact_json(self, storage, sample_graph):
        result = storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        content = result['graph_json'].read_text()
        # Compact JSON ما فيه indent
        assert '\n' not in content.split('nodes')[0]  # header should be on one line

    def test_save_preserves_nodes(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        assert g.number_of_nodes() == 10

    def test_save_preserves_edges(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        assert g.number_of_edges() == 8

    def test_save_preserves_embeddings(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        emb = g.nodes['n0'].get('embedding')
        assert emb is not None
        assert len(emb) == 64

    def test_save_preserves_content(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        assert g.nodes['n0'].get('content') == 'Content 0'

    def test_save_preserves_metadata(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        assert g.nodes['n0'].get('metadata', {}).get('source') == 'file0.txt'


# ============================================================
# load() tests
# ============================================================

class TestLoad:
    def test_load_existing_graph(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        g = storage.load('test')
        assert isinstance(g, nx.Graph)

    def test_load_missing_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.load('nonexistent')


# ============================================================
# load_communities() tests
# ============================================================

class TestLoadCommunities:
    def test_load_existing(self, storage, sample_graph):
        communities = {'communities': {'0': ['n0', 'n1']}, 'num_communities': 1, 'modularity': 0.5}
        storage.save(sample_graph, communities, 'test', generate_report=False)
        loaded = storage.load_communities('test')
        assert loaded['num_communities'] == 1

    def test_load_missing_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.load_communities('nonexistent')


# ============================================================
# list_projects() tests
# ============================================================

class TestListProjects:
    def test_empty_dir(self, storage):
        projects = storage.list_projects()
        assert projects == []

    def test_one_project(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'proj1', generate_report=False)
        projects = storage.list_projects()
        assert projects == ['proj1']

    def test_multiple_projects_sorted(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'zebra', generate_report=False)
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'alpha', generate_report=False)
        projects = storage.list_projects()
        assert projects == ['alpha', 'zebra']

    def test_ignores_hidden_dirs(self, storage):
        (Path(storage.graphs_dir) / '.hidden').mkdir()
        projects = storage.list_projects()
        assert '.hidden' not in projects


# ============================================================
# delete_project() tests
# ============================================================

class TestDeleteProject:
    def test_delete_existing(self, storage, sample_graph):
        storage.save(sample_graph, {'communities': {}, 'num_communities': 0, 'modularity': 0}, 'test', generate_report=False)
        result = storage.delete_project('test')
        assert result is True
        assert not (storage.graphs_dir / 'test' / 'graph.json').exists()

    def test_delete_nonexistent(self, storage):
        result = storage.delete_project('nonexistent')
        assert result is False


# ============================================================
# _format_size() tests
# ============================================================

class TestFormatSize:
    def test_bytes(self, storage):
        assert 'B' in storage._format_size(500)

    def test_kb(self, storage):
        assert 'KB' in storage._format_size(2048)

    def test_mb(self, storage):
        assert 'MB' in storage._format_size(2_000_000)


# ============================================================
# Backward compatibility with old indent=2 format
# ============================================================

class TestBackwardCompatibility:
    def test_loads_old_format(self, storage, sample_graph):
        # Save in old format (indent=2)
        graph_path = storage.graphs_dir / 'old' / 'graph.json'
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_data = nx.node_link_data(sample_graph)
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # Load should work
        g = storage.load('old')
        assert g.number_of_nodes() == 10
