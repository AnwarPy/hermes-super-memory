"""Community Detector — اكتشاف المجموعات الدلالية

يدعم:
- Leiden Algorithm (مُوصى به — أسرع وأدق من Louvain)
- Louvain Algorithm (fallback)
- Connected Components (fallback للرسومات الكبيرة جداً)
- إحصائيات المجموعات
- حدود أمان للحواف
"""

import networkx as nx
from typing import Dict, List, Any, Optional


MAX_EDGES_FOR_COMMUNITY = 50000  # فوق هذا → connected components فقط


class CommunityDetector:
    """كاشف المجموعات الدلالية"""
    
    def __init__(self, algorithm: str = "leiden"):
        """
        Args:
            algorithm: خوارزمية الاكتشاف ("leiden" أو "louvain")
        """
        self.algorithm = algorithm.lower()
        
        if self.algorithm not in ["leiden", "louvain"]:
            raise ValueError(f"خوارزمية غير مدعومة: {algorithm}. استخدم 'leiden' أو 'louvain'")
    
    def detect(
        self,
        graph: nx.Graph,
        weight: str = "weight",
        resolution: float = 1.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        اكتشاف المجموعات
        
        Args:
            graph: الرسم المعرفي
            weight: اسم حقل الوزن في الحواف
            resolution: معامل الدقة (أعلى = مجتمعات أصغر)
            seed: بذرة عشوائية للاستقرار
        
        Returns:
            dict يحتوي على:
                - communities: {community_id: [node_ids]}
                - modularity: درجة جودة التقسيم
                - num_communities: عدد المجتمعات
                - node_to_community: {node_id: community_id}
        """
        num_edges = graph.number_of_edges()
        
        # حماية: رسومات كبيرة جداً → connected components فقط
        if num_edges > MAX_EDGES_FOR_COMMUNITY:
            print(f"  ⚠️ رسم كبير جداً ({num_edges:,} حافة) — استخدام connected components")
            return self._detect_connected_components(graph)
        
        # حاول Leiden أولاً (أفضل من Louvain)
        if self.algorithm == "leiden":
            try:
                return self._detect_leiden(graph, weight, resolution, seed)
            except Exception as e:
                print(f"  ⚠️ Leiden فشل ({e}) — fallback لـ Louvain")
        
        # Louvain
        return self._detect_louvain(graph, weight, seed)
    
    def _detect_leiden(
        self,
        graph: nx.Graph,
        weight: str,
        resolution: float,
        seed: int,
    ) -> Dict[str, Any]:
        """اكتشاف باستخدام Leiden Algorithm"""
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            print("تحذير: leidenalg أو igraph غير مثبت، جاري استخدام Louvain")
            return self._detect_louvain(graph, weight, seed)
        
        # تحويل من NetworkX إلى igraph
        ig_graph = ig.Graph.from_networkx(graph)
        
        # التأكد من وجود أوزان
        edge_weights = [e[2].get(weight, 1.0) for e in graph.edges(data=True)]
        if edge_weights:
            ig_graph.es[weight] = edge_weights
        
        # اكتشاف المجتمعات - leidenalg 0.10+
        try:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights=ig_graph.es[weight] if weight in ig_graph.es.attributes() else None,
                n_iterations=-1,
                seed=seed,
            )
        except Exception as e:
            # محاولة بدون weights لو فشلت
            if 'weight' in str(e).lower():
                partition = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.ModularityVertexPartition,
                    n_iterations=-1,
                    seed=seed,
                )
            else:
                raise
        
        # تحويل النتائج
        communities = {}
        node_to_community = {}
        
        for node_idx, comm_id in enumerate(partition.membership):
            node_id = list(graph.nodes())[node_idx] if node_idx < len(list(graph.nodes())) else f"node_{node_idx}"
            
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node_id)
            node_to_community[node_id] = comm_id
        
        # حساب Modularity
        modularity = partition.quality() if callable(getattr(partition, 'quality', None)) else partition.quality
        
        return {
            "communities": communities,
            "modularity": float(modularity),
            "num_communities": len(communities),
            "node_to_community": node_to_community,
            "algorithm": "leiden",
        }
    
    def _detect_louvain(
        self,
        graph: nx.Graph,
        weight: str,
        seed: int,
    ) -> Dict[str, Any]:
        """اكتشاف باستخدام Louvain Algorithm"""
        try:
            import community as community_louvain
        except ImportError:
            raise ImportError("python-louvain غير مثبت. قم بتثبيته: pip install python-louvain")
        
        # اكتشاف المجتمعات
        partition = community_louvain.best_partition(
            graph,
            weight=weight,
            random_state=seed,
        )
        
        # تحويل النتائج
        communities = {}
        node_to_community = partition
        
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # حساب Modularity
        modularity = community_louvain.modularity(partition, graph)
        
        return {
            "communities": communities,
            "modularity": float(modularity),
            "num_communities": len(communities),
            "node_to_community": node_to_community,
            "algorithm": "louvain",
        }
    
    def _detect_connected_components(
        self,
        graph: nx.Graph,
    ) -> Dict[str, Any]:
        """اكتشاف باستخدام connected components (سريع جداً للرسومات الكبيرة)"""
        components = list(nx.connected_components(graph))
        
        communities = {}
        node_to_community = {}
        
        for comm_id, component in enumerate(components):
            comm_id_str = str(comm_id)
            communities[comm_id_str] = list(component)
            for node in component:
                node_to_community[node] = comm_id
        
        return {
            "communities": communities,
            "modularity": 0.0,  # لا modularity للـ connected components
            "num_communities": len(communities),
            "node_to_community": node_to_community,
            "algorithm": "connected_components",
        }
    
    def get_community_stats(
        self,
        graph: nx.Graph,
        communities: Dict[int, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        إحصائيات لكل مجتمع
        
        Args:
            graph: الرسم المعرفي
            communities: قاموس المجتمعات
        
        Returns:
            قائمة إحصائيات لكل مجتمع
        """
        stats = []
        
        for comm_id, nodes in communities.items():
            # استخراج المجتمع الفرعي
            subgraph = graph.subgraph(nodes)
            
            stat = {
                "community_id": comm_id,
                "num_nodes": len(nodes),
                "num_edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph),
                "avg_degree": sum(dict(subgraph.degree()).values()) / len(nodes) if nodes else 0,
            }
            stats.append(stat)
        
        # ترتيب حسب حجم المجتمع
        stats.sort(key=lambda x: x["num_nodes"], reverse=True)
        
        return stats
