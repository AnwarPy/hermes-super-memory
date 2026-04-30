"""Graphify Engine — المحرك الرئيسي للرسم المعرفي

الوظائف:
- تنسيق جميع المكونات (Loader, Splitter, Embedding, Builder, Detector, Storage)
- فهرسة مجلدات كاملة
- بحث دلالي
- تحديث تدريجي
- حدود أمان (max_nodes, max_file_size)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import networkx as nx

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embedding_model import EmbeddingModel
from .graph_builder import KnowledgeGraphBuilder
from .community_detector import CommunityDetector
from .graph_storage import GraphStorage


# حدود أمان
MAX_NODES_HARD = 2500      # إيقاف كامل
MAX_NODES_WARNING = 2000   # تحذير
MAX_EDGES_WARNING = 10000  # تحذير قبل Louvain


class GraphifyEngine:
    """المحرك الرئيسي للرسم المعرفي"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: تكوين المحرك
        """
        self.config = config or {}
        
        # إعدادات
        self.graphs_dir = self.config.get("graphs_dir", "~/.hermes/graphs")
        self.embedding_model_name = self.config.get("embedding_model", "BAAI/bge-m3")
        self.device = self.config.get("device", "auto")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.chunk_size = self.config.get("chunk_size", 512)  # 512 أفضل للعربية
        self.chunk_overlap = self.config.get("chunk_overlap", 96)
        self.community_algorithm = self.config.get("community_algorithm", "leiden")
        
        # تهيئة المكونات
        print("جاري تهيئة Graphify Engine...")
        
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.embedding = EmbeddingModel(
            model_name=self.embedding_model_name,
            device=self.device,
        )
        self.builder = KnowledgeGraphBuilder(self.embedding)
        self.detector = CommunityDetector(algorithm=self.community_algorithm)
        self.storage = GraphStorage(self.graphs_dir)
        
        print("✓ Graphify Engine جاهز")
    
    def index_directory(
        self,
        path: str,
        project_name: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        فهرسة مجلد كامل
        
        Args:
            path: مسار المجلد
            project_name: اسم المشروع (افتراضي: اسم المجلد)
            patterns: أنماط الملفات
            recursive: بحث متداخل
            reindex: إعادة فهرسة كاملة
        
        Returns:
            تقرير الفهرسة
        """
        start_time = datetime.now()
        path = Path(path).expanduser()
        
        if not path.exists():
            raise FileNotFoundError(f"المجلد غير موجود: {path}")
        
        # اسم المشروع
        if project_name is None:
            project_name = path.name
        
        print(f"\n{'='*60}")
        print(f"فهرسة مشروع: {project_name}")
        print(f"المسار: {path}")
        print(f"{'='*60}\n")
        
        # التحقق من الفهرسة السابقة
        if not reindex:
            try:
                existing_graph = self.storage.load(project_name)
                print(f"⚠️ رسم موجود مسبقاً ({existing_graph.number_of_nodes()} عقدة)")
                print("  استخدم reindex=True لإعادة الفهرسة الكاملة\n")
            except FileNotFoundError:
                pass  # لا يوجد رسم سابق
        
        # 1. تحميل المستندات
        print("المرحلة 1/6: تحميل المستندات...")
        docs = self.loader.load_directory(str(path), patterns, recursive)
        print(f"  ✓ تم تحميل {len(docs)} وثيقة\n")
        
        if not docs:
            return {
                "status": "no_documents",
                "project_name": project_name,
                "message": "لم يتم العثور على مستندات",
            }
        
        # 2. تقسيم النصوص
        print("المرحلة 2/6: تقسيم النصوص...")
        split_docs = []
        for doc in docs:
            file_type = Path(doc.metadata.get("source", "")).suffix.lower().replace(".", "")
            chunks = self.splitter.split([doc], file_type or None)
            split_docs.extend(chunks)
        print(f"  ✓ تم تقسيم إلى {len(split_docs)} قطعة\n")
        
        # 3. بناء الرسم
        print("المرحلة 3/6: بناء الرسم المعرفي...")
        self.builder = KnowledgeGraphBuilder(self.embedding)
        node_ids = self.builder.add_nodes_from_docs(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in split_docs],
        )
        
        # تحقق: حد أقصى للعقد
        num_nodes = len(node_ids)
        if num_nodes > MAX_NODES_HARD:
            raise ValueError(
                f"عدد العقد {num_nodes:,} يتجاوز الحد الأقصى ({MAX_NODES_HARD:,}). "
                f"المجلد كبير جداً — قسمه أو استخدم patterns أضيق."
            )
        if num_nodes > MAX_NODES_WARNING:
            print(f"  ⚠️ تحذير: {num_nodes:,} عقدة (أقرب من الحد الأقصى {MAX_NODES_HARD:,})")
        
        print(f"  ✓ تمت إضافة {num_nodes} عقدة\n")
        
        # 4. إضافة الحواف
        print("المرحلة 4/6: إضافة الحواف الدلالية...")
        edges_added = self.builder.add_edges_by_similarity(
            threshold=self.similarity_threshold,
            batch_size=100,
        )
        
        # تحقق: تحذير إذا الحواف كثيرة
        if edges_added > MAX_EDGES_WARNING:
            print(f"  ⚠️ تحذير: {edges_added:,} حافة — كثافة عالية قد تؤثر على الأداء")
        
        print(f"  ✓ تمت إضافة {edges_added} حافة\n")
        
        # 5. اكتشاف المجتمعات
        print("المرحلة 5/6: اكتشاف المجموعات الدلالية...")
        communities = self.detector.detect(self.builder.graph)
        print(f"  ✓ تم اكتشاف {communities['num_communities']} مجتمعات")
        print(f"    Modularity Score: {communities['modularity']:.4f}\n")
        
        # 6. الحفظ
        print("المرحلة 6/6: حفظ الرسم...")
        saved_files = self.storage.save(
            self.builder.graph,
            communities,
            project_name,
            generate_report=True,
        )
        print(f"  ✓ تم الحفظ في {self.storage.graphs_dir / project_name}\n")
        
        # تقرير نهائي
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats = self.builder.get_stats()
        
        report = {
            "status": "success",
            "project_name": project_name,
            "path": str(path),
            "duration_seconds": duration,
            "stats": {
                "documents_loaded": len(docs),
                "chunks_created": len(split_docs),
                "nodes_added": stats["num_nodes"],
                "edges_added": stats["num_edges"],
                "density": stats["density"],
                "avg_degree": stats["avg_degree"],
                "communities_detected": communities["num_communities"],
                "modularity_score": communities["modularity"],
            },
            "output": {
                "graph_json": str(saved_files["graph_json"]),
                "communities_json": str(saved_files["communities_json"]),
                "report_md": str(saved_files["report_md"]),
                "metadata_json": str(saved_files["metadata_json"]),
            },
        }
        
        print(f"{'='*60}")
        print(f"✅ اكتملت الفهرسة في {duration:.2f} ثانية")
        print(f"{'='*60}\n")
        
        return report
    
    def search_semantic(
        self,
        query: str,
        project_name: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        بحث دلالي في الرسم المعرفي
        
        Args:
            query: نص البحث
            project_name: اسم المشروع
            top_k: عدد النتائج
            min_similarity: الحد الأدنى للتشابه
        
        Returns:
            قائمة النتائج
        """
        # تحميل الرسم
        try:
            graph = self.storage.load(project_name)
        except FileNotFoundError:
            return []
        
        # تضمين الاستعلام
        query_embedding = self.embedding.embed_query(query)
        
        # البحث عن أقرب العقد
        results = []
        for node_id in graph.nodes():
            node_embedding = graph.nodes[node_id].get("embedding")
            if node_embedding is None:
                continue
            
            # حساب التشابه
            import numpy as np
            v1 = np.array(query_embedding)
            v2 = np.array(node_embedding)
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            if similarity >= min_similarity:
                results.append({
                    "node_id": node_id,
                    "content": graph.nodes[node_id].get("content", ""),
                    "similarity": float(similarity),
                    "metadata": graph.nodes[node_id].get("metadata", {}),
                    "type": graph.nodes[node_id].get("type", "unknown"),
                })
        
        # ترتيب حسب التشابه
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def get_graph_stats(self, project_name: str) -> Dict[str, Any]:
        """إحصائيات الرسم"""
        try:
            graph = self.storage.load(project_name)
            communities = self.storage.load_communities(project_name)
            
            return {
                "project_name": project_name,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "num_communities": communities.get("num_communities", 0),
                "modularity": communities.get("modularity", 0),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_projects(self) -> List[str]:
        """سرد المشاريع المفهرسة"""
        return self.storage.list_projects()
