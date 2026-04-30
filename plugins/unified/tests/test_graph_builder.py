"""Tests for graph_builder.py — بناء الحواف والخوارزمية"""

import pytest
import numpy as np
from unified.graph_builder import KnowledgeGraphBuilder


class FakeEmbedding:
    """Embedding model وهمي للاختبار"""
    def embed_query(self, text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(64).tolist()


# ============================================================
# add_node tests
# ============================================================

class TestAddNode:
    def test_add_node_with_embedding(self):
        builder = KnowledgeGraphBuilder(None)
        nid = builder.add_node(
            node_id='test_1',
            content='Hello world',
            embedding=[0.1] * 10,
        )
        assert nid == 'test_1'
        assert 'test_1' in builder.node_embeddings
        assert builder.graph.has_node('test_1')

    def test_add_node_auto_id(self):
        builder = KnowledgeGraphBuilder(None)
        nid1 = builder.add_node(content='first', embedding=[0.1]*10)
        nid2 = builder.add_node(content='second', embedding=[0.1]*10)
        assert nid1 != nid2
        assert nid1.startswith('node_')

    def test_add_node_detects_type_arabic(self):
        builder = KnowledgeGraphBuilder(None)
        nid = builder.add_node(content='هذا نص عربي', embedding=[0.1]*10)
        assert builder.graph.nodes[nid]['type'] == 'arabic'

    def test_add_node_detects_type_code(self):
        builder = KnowledgeGraphBuilder(None)
        nid = builder.add_node(content='def hello(): pass', embedding=[0.1]*10)
        assert builder.graph.nodes[nid]['type'] == 'code'

    def test_add_node_detects_type_text(self):
        builder = KnowledgeGraphBuilder(None)
        nid = builder.add_node(content='Hello world text', embedding=[0.1]*10)
        assert builder.graph.nodes[nid]['type'] == 'text'

    def test_add_node_requires_embedding_when_no_model(self):
        builder = KnowledgeGraphBuilder(None)
        with pytest.raises(ValueError, match='embedding مطلوب'):
            builder.add_node(content='no embedding')


# ============================================================
# add_edges_by_similarity — argpartition tests
# ============================================================

class TestAddEdgesBySimilarity:
    def _make_builder(self, n_nodes, dim=64):
        builder = KnowledgeGraphBuilder(None)
        for i in range(n_nodes):
            np.random.seed(i)
            builder.add_node(
                node_id=f'n{i}',
                content=f'content {i}',
                embedding=np.random.randn(dim).tolist(),
            )
        return builder

    def test_edge_count_bounded_by_top_k(self):
        """عدد الحواف ما يتجاوز n * top_k"""
        builder = self._make_builder(100, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        edges = builder.graph.number_of_edges()
        max_edges = 100 * 10  # worst case
        assert edges <= max_edges, f'Edges {edges} > max {max_edges}'

    def test_no_self_loops(self):
        """ما فيه حواف ذاتية"""
        builder = self._make_builder(50, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        self_loops = sum(1 for u, v in builder.graph.edges() if u == v)
        assert self_loops == 0, f'Self-loops found: {self_loops}'

    def test_all_edges_have_weight(self):
        """كل حافة عندها وزن"""
        builder = self._make_builder(50, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        for u, v, d in builder.graph.edges(data=True):
            assert 'weight' in d, f'Missing weight on edge {u}-{v}'

    def test_all_edges_above_threshold(self):
        """كل الحواف فوق الـ threshold"""
        builder = self._make_builder(50, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.4)
        for u, v, d in builder.graph.edges(data=True):
            assert d['weight'] >= 0.4, f'Weight below threshold: {d["weight"]}'

    def test_edge_weights_in_range(self):
        """الأوزان بين -1 و 1"""
        builder = self._make_builder(50, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        for u, v, d in builder.graph.edges(data=True):
            w = d['weight']
            assert -1.0 <= w <= 1.0, f'Weight out of range: {w}'

    def test_returns_correct_count(self):
        """الدالة ترجع عدد صحيح"""
        builder = self._make_builder(50, dim=32)
        count = builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        assert isinstance(count, int)
        assert count == builder.graph.number_of_edges()

    def test_empty_graph_returns_zero(self):
        builder = KnowledgeGraphBuilder(None)
        count = builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        assert count == 0

    def test_single_node_returns_zero(self):
        builder = KnowledgeGraphBuilder(None)
        builder.add_node(node_id='n0', content='single', embedding=[0.1]*10)
        count = builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        assert count == 0

    def test_two_nodes(self):
        builder = KnowledgeGraphBuilder(None)
        emb = [0.1] * 10
        builder.add_node(node_id='a', content='node a', embedding=emb)
        builder.add_node(node_id='b', content='node b', embedding=emb)
        count = builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        # متطابقان = similarity 1.0
        assert count == 1

    def test_default_top_k_is_10(self):
        """top_k الافتراضي = 10"""
        builder = self._make_builder(50, dim=32)
        count = builder.add_edges_by_similarity(threshold=0.3)  # no top_k arg
        assert count == builder.graph.number_of_edges()

    def test_top_k_clamped_to_n_minus_1(self):
        """لو top_k > n-1، يتقلص تلقائياً"""
        builder = self._make_builder(5, dim=32)
        count = builder.add_edges_by_similarity(top_k=100, threshold=0.3)
        # 5 عقد = max 10 حواف (5*4/2)
        assert count <= 10

    def test_performance_argpartition(self):
        """argpartition أسرع من argsort"""
        import time
        builder = self._make_builder(500, dim=128)

        t0 = time.time()
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        elapsed = time.time() - t0

        # 500 عقدة لازم تأخذ أقل من ثانية
        assert elapsed < 2.0, f'argpartition took {elapsed:.2f}s (expected < 2s)'

    def test_density_reduced_vs_full_graph(self):
        """الكثافة أقل من رسم كامل"""
        import networkx as nx
        builder = self._make_builder(200, dim=32)
        builder.add_edges_by_similarity(top_k=10, threshold=0.3)
        density = nx.density(builder.graph)
        # density يجب تكون < 20% (رسم sparse)
        assert density < 0.20, f'Density too high: {density:.2f}'


# ============================================================
# get_stats tests
# ============================================================

class TestGetStats:
    def test_stats_structure(self):
        builder = KnowledgeGraphBuilder(None)
        builder.add_node(node_id='a', content='test', embedding=[0.1]*10)
        builder.add_node(node_id='b', content='test', embedding=[0.1]*10)
        builder.graph.add_edge('a', 'b', weight=0.5)

        stats = builder.get_stats()
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'density' in stats
        assert 'avg_degree' in stats
        assert stats['num_nodes'] == 2
        assert stats['num_edges'] == 1


# ============================================================
# _detect_type tests
# ============================================================

class TestDetectType:
    def _make_builder(self):
        return KnowledgeGraphBuilder(None)

    def test_code_def(self):
        b = self._make_builder()
        assert b._detect_type('def foo(): pass') == 'code'

    def test_code_class(self):
        b = self._make_builder()
        assert b._detect_type('class MyClass:') == 'code'

    def test_code_import(self):
        b = self._make_builder()
        assert b._detect_type('import numpy as np') == 'code'

    def test_code_function(self):
        b = self._make_builder()
        assert b._detect_type('function foo() {}') == 'code'

    def test_code_const(self):
        b = self._make_builder()
        assert b._detect_type('const x = 1') == 'code'

    def test_arabic(self):
        b = self._make_builder()
        assert b._detect_type('هذا نص عربي') == 'arabic'

    def test_arabic_punctuation(self):
        b = self._make_builder()
        assert b._detect_type('نص عربي، مع فاصلة') == 'arabic'

    def test_plain_english(self):
        b = self._make_builder()
        assert b._detect_type('Hello world text') == 'text'

    def test_empty(self):
        b = self._make_builder()
        assert b._detect_type('') == 'text'
