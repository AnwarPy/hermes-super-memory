"""Graph Builder — بناء الرسم المعرفي باستخدام NetworkX

الوظائف:
- إضافة عقد مع التضمينات
- إضافة حواف بناءً على التشابه الدلالي (top_k إجباري)
- كشف نوع العقدة تلقائياً
- معالجة دفعات كبيرة (Batch Processing)
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class KnowledgeGraphBuilder:
    """باني الرسوم المعرفية"""
    
    # قيم افتراضية آمنة
    DEFAULT_TOP_K = 10        # كل عقدة تتصل بأقرب 10 عقد
    DEFAULT_MIN_SIMILARITY = 0.4  # حد أدنى للتشابه (فلتر ضوضاء)
    
    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: نموذج التضمين (EmbeddingModel)
        """
        self.embeddings = embedding_model
        self.graph = nx.Graph()
        self.node_embeddings = {}  # cache للتضمينات
        self._node_counter = 0
    
    def add_node(
        self,
        node_id: Optional[str] = None,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        إضافة عقدة للرسم
        
        Args:
            node_id: معرف العقدة (توليد تلقائي إذا لم يُحدد)
            content: محتوى العقدة (نص)
            metadata: بيانات وصفية
            embedding: متجه التضمين (اختياري، يُستخرج إذا لم يُقدم)
        
        Returns:
            node_id: معرف العقدة المضافة
        """
        if node_id is None:
            node_id = f"node_{self._node_counter}"
            self._node_counter += 1
        
        # استخراج التضمين إذا لم يُقدم
        if embedding is None and self.embeddings:
            # تطبيع النص العربي قبل التضمين — للاتساق مع وقت البحث
            # (انظر arabic_normalizer.normalize_query في __init__.py:569)
            normalize_query = None
            is_arabic = None
            try:
                from .arabic_normalizer import normalize_query, is_arabic
            except (ImportError, ValueError):
                try:
                    from arabic_normalizer import normalize_query, is_arabic  # type: ignore
                except ImportError:
                    pass
            if normalize_query is not None and is_arabic(content):
                embed_text = normalize_query(content)
            else:
                embed_text = content
            embedding = self.embeddings.embed_query(embed_text)
        elif embedding is None:
            raise ValueError("embedding مطلوب إذا لم يكن embedding_model مُهيأ")
        
        # كشف نوع العقدة
        node_type = self._detect_type(content)
        
        # إضافة العقدة
        self.graph.add_node(
            node_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            type=node_type,
            created_at=datetime.now().isoformat(),
        )
        
        # حفظ التضمين في cache
        self.node_embeddings[node_id] = embedding
        
        return node_id
    
    def add_nodes_from_docs(
        self,
        documents: List[dict],
        batch_size: int = 100,
    ) -> List[str]:
        """
        إضافة عقد من قائمة مستندات
        
        Args:
            documents: قائمة المستندات (كل مستند: {page_content, metadata})
            batch_size: حجم الدفعة
        
        Returns:
            قائمة معرفات العقد المضافة
        """
        node_ids = []
        
        # استخراج النصوص — مع تطبيع للنصوص العربية للاتساق وقت البحث
        normalize_query = None
        is_arabic = None
        try:
            from .arabic_normalizer import normalize_query, is_arabic
        except (ImportError, ValueError):
            try:
                from arabic_normalizer import normalize_query, is_arabic  # type: ignore
            except ImportError:
                pass
        
        raw_texts = [doc.get("page_content", "") for doc in documents]
        if normalize_query is not None:
            texts = [
                normalize_query(t) if (t and is_arabic(t)) else t
                for t in raw_texts
            ]
        else:
            texts = raw_texts
        
        # استخراج دفعات من التضمينات (على النص المطبَّع)
        print(f"جاري استخراج تضمينات {len(texts)} وثيقة...")
        all_embeddings = self.embeddings.embed_documents(texts, batch_size=batch_size)
        
        # إضافة العقد — نخزّن المحتوى الأصلي للعرض، والتضمين على المطبَّع
        for i, (doc, emb) in enumerate(zip(documents, all_embeddings)):
            node_id = self.add_node(
                content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {}),
                embedding=emb,
            )
            node_ids.append(node_id)
            
            # تقدم كل 100 عقدة
            if (i + 1) % 100 == 0:
                print(f"  تمت إضافة {i + 1}/{len(documents)} عقدة")
        
        print(f"✓ تمت إضافة {len(node_ids)} عقدة")
        return node_ids
    
    def add_edges_by_similarity(
        self,
        threshold: float = 0.4,
        batch_size: int = 100,
        top_k: Optional[int] = None,
    ) -> int:
        """
        إضافة حواف بناءً على التشابه الدلالي باستخدام argpartition (O(n·k))
        
        خوارزمية جديدة:
        1. لكل عقدة: أجد أقرب top_k عقد باستخدام np.argpartition (O(n) بدل O(n²))
        2. فلتر: فقط الحواف فوق threshold تُضاف
        3. deduplication: كل حافة تضاف مرة واحدة بأعلى وزن
        
        Args:
            threshold: حد أدنى للتشابه (فلتر ضوضاء، ليس limiter)
            batch_size: حجم الدفعة لحساب embeddings (للتوافق فقط)
            top_k: عدد أقرب العقد لكل عقدة (افتراضي: 10)
        
        Returns:
            عدد الحواف المضافة
        """
        node_ids = list(self.node_embeddings.keys())
        n = len(node_ids)
        
        if n < 2:
            return 0
        
        if top_k is None:
            top_k = self.DEFAULT_TOP_K
        
        # Clamp top_k to avoid edges to all nodes
        top_k = min(top_k, n - 1)
        
        print(f"جاري إضافة الحواف لـ {n} عقدة (top_k={top_k}, min_sim={threshold})...")
        
        # تحويل embeddings إلى numpy matrix (n, dim)
        emb_matrix = np.array(
            [self.node_embeddings[nid] for nid in node_ids],
            dtype=np.float32,
        )
        
        # تطبيع embeddings لحساب cosine similarity عبر dot product
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # تجنب القسمة على صفر
        emb_normalized = emb_matrix / norms
        
        # مصفوفة التشابه الكاملة: (n, n)
        # cosine similarity = dot product بعد التطبيع
        similarity_matrix = emb_normalized @ emb_normalized.T
        
        # بناء الحواف باستخدام argpartition
        edge_weights: Dict[tuple, float] = {}  # (i, j) → max weight
        edges_added = 0
        
        for i in range(n):
            # استبعاد self-similarity
            sim_row = similarity_matrix[i].copy()
            sim_row[i] = -1.0
            
            # argpartition: O(n) بدلاً من O(n log n)
            if top_k < n - 1:
                top_indices = np.argpartition(sim_row, -top_k)[-top_k:]
            else:
                top_indices = np.arange(n)
            
            for j in top_indices:
                sim = float(sim_row[j])
                
                # فلتر الحد الأدنى
                if sim < threshold:
                    continue
                
                # Deduplication: استخدم (min, max) لتجنب التكرار
                edge = (min(i, j), max(i, j))
                if edge not in edge_weights or sim > edge_weights[edge]:
                    if edge not in edge_weights:
                        edges_added += 1
                    edge_weights[edge] = sim
        
        # إضافة الحواف للرسم
        for (i, j), weight in edge_weights.items():
            self.graph.add_edge(
                node_ids[i],
                node_ids[j],
                weight=weight,
                type="semantic",
                created_at=datetime.now().isoformat(),
            )
        
        print(f"✓ تمت إضافة {edges_added} حافة")
        return edges_added
    
    def _detect_type(self, content: str) -> str:
        """
        كشف نوع العقدة تلقائياً
        
        Args:
            content: محتوى العقدة
        
        Returns:
            نوع العقدة (code, arabic, text)
        """
        content = content.strip()
        
        # كود
        if content.startswith(("def ", "class ", "import ", "from ", "function ", "const ", "let ", "var ")):
            return "code"
        
        # عربي (وجود أحرف عربية)
        arabic_chars = {"؟", "!", "۔", "،", "؛", "ء", "آ", "أ", "إ", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"}
        if any(c in content for c in arabic_chars):
            return "arabic"
        
        # نص عادي
        return "text"
    
    def get_stats(self) -> dict:
        """إحصائيات الرسم"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
        }
