"""Tests for community_detector.py — اكتشاف المجموعات"""

import pytest
import networkx as nx
from unified.community_detector import CommunityDetector, MAX_EDGES_FOR_COMMUNITY


# ============================================================
# detect() — Leiden path
# ============================================================

class TestDetectLeiden:
    """اختبار اكتشاف المجموعات بـ Leiden"""

    def _make_clustered_graph(self):
        """رسم بـ 3 مجتمعات واضحة"""
        g = nx.Graph()
        # Community 1: nodes 0-4 (complete graph)
        for i in range(5):
            for j in range(i+1, 5):
                g.add_edge(f'c1_{i}', f'c1_{j}', weight=0.9)
        # Community 2: nodes 5-9
        for i in range(5, 10):
            for j in range(i+1, 10):
                g.add_edge(f'c2_{i}', f'c2_{j}', weight=0.9)
        # Community 3: nodes 10-14
        for i in range(10, 15):
            for j in range(i+1, 15):
                g.add_edge(f'c3_{i}', f'c3_{j}', weight=0.9)
        # Weak inter-community edges
        g.add_edge('c1_0', 'c2_5', weight=0.1)
        g.add_edge('c2_9', 'c3_10', weight=0.1)
        return g

    def test_detect_returns_dict(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        for key in ['communities', 'modularity', 'num_communities', 'node_to_community', 'algorithm']:
            assert key in result, f'Missing key: {key}'

    def test_all_nodes_assigned(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        assert len(result['node_to_community']) == g.number_of_nodes()

    def test_algorithm_is_leiden(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        assert result['algorithm'] == 'leiden'

    def test_modularity_is_float(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        assert isinstance(result['modularity'], float)

    def test_detects_multiple_communities(self):
        g = self._make_clustered_graph()
        det = CommunityDetector('leiden')
        result = det.detect(g)
        # Must detect at least 2 communities
        assert result['num_communities'] >= 2


# ============================================================
# detect() — Louvain path
# ============================================================

class TestDetectLouvain:
    """اختبار اكتشاف المجموعات بـ Louvain"""

    def _make_simple_graph(self):
        g = nx.Graph()
        for i in range(6):
            for j in range(i+1, 6):
                g.add_edge(f'a{i}', f'a{j}', weight=0.8)
        for i in range(6, 12):
            for j in range(i+1, 12):
                g.add_edge(f'b{i}', f'b{j}', weight=0.8)
        g.add_edge('a0', 'b6', weight=0.1)
        return g

    def test_louvain_detect_returns_dict(self):
        g = self._make_simple_graph()
        det = CommunityDetector('louvain')
        result = det.detect(g)
        assert isinstance(result, dict)

    def test_louvain_algorithm_name(self):
        g = self._make_simple_graph()
        det = CommunityDetector('louvain')
        result = det.detect(g)
        assert result['algorithm'] == 'louvain'

    def test_louvain_all_nodes_assigned(self):
        g = self._make_simple_graph()
        det = CommunityDetector('louvain')
        result = det.detect(g)
        assert len(result['node_to_community']) == g.number_of_nodes()


# ============================================================
# Connected components fallback
# ============================================================

class TestConnectedComponents:
    """اختبار fallback للـ connected components"""

    def test_single_component(self):
        g = nx.Graph()
        for i in range(10):
            g.add_edge(f'n{i}', f'n{i+1}', weight=0.5)
        det = CommunityDetector('leiden')
        result = det._detect_connected_components(g)
        assert result['num_communities'] == 1
        assert result['algorithm'] == 'connected_components'
        assert result['modularity'] == 0.0

    def test_multiple_components(self):
        g = nx.Graph()
        g.add_edge('a1', 'a2', weight=0.5)
        g.add_edge('a2', 'a3', weight=0.5)
        g.add_edge('b1', 'b2', weight=0.5)
        det = CommunityDetector('leiden')
        result = det._detect_connected_components(g)
        assert result['num_communities'] == 2

    def test_all_nodes_assigned(self):
        g = nx.Graph()
        g.add_node('isolated')
        g.add_edge('a', 'b', weight=0.5)
        det = CommunityDetector('leiden')
        result = det._detect_connected_components(g)
        assert len(result['node_to_community']) == 3


# ============================================================
# MAX_EDGES_FOR_COMMUNITY guard
# ============================================================

class TestMaxEdgesGuard:
    """اختبار حماية الحواف الكبيرة"""

    def test_large_graph_uses_connected_components(self):
        """رسومات كبيرة جداً تستخدم connected components"""
        g = nx.Graph()
        n = 500  # 500 nodes → C(500,2) = 124,750 max edges
        # Add enough edges to exceed threshold
        for i in range(n):
            for j in range(i+1, min(i+200, n)):
                g.add_edge(f'n{i}', f'n{j}', weight=0.5)

        assert g.number_of_edges() > MAX_EDGES_FOR_COMMUNITY, \
            f'Need > {MAX_EDGES_FOR_COMMUNITY} edges, got {g.number_of_edges()}'

        det = CommunityDetector('leiden')
        result = det.detect(g)
        assert result['algorithm'] == 'connected_components'


# ============================================================
# Constants
# ============================================================

class TestConstants:
    def test_max_edges_defined(self):
        assert MAX_EDGES_FOR_COMMUNITY > 0

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match='خوارزمية غير مدعومة'):
            CommunityDetector('invalid')
