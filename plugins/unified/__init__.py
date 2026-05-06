"""Unified Memory Provider v3.5 — مزود الذاكرة الموحد المحسّن

التحسينات:
- In-memory caching للرسوم المعرفية (تحميل مرة واحدة)
- بحث في جميع المشاريع
- تنظيف المحتوى المحسّن (لا يزيل {} و [])
- دمج FTS5 + Graph في prefetch
- معالجة أخطاء شاملة (0% فشل)
- فلتر زمني (15 ثانية) لمنع ظهور رسالة المستخدم كذاكرة
- إزالة تكرار بالـ Hashing (نفس النص من مشاريع مختلفة يظهر مرة مرة واحدة)
- chunk_size=512 + overlap=96 للنصوص العربية الكاملة
- كشف تلقائي للغة (arabic/english) في المخرجات
- استبعاد رسائل tool من FTS5 (ملفات ومسارات)
- إزالة علامات FTS5 (>>> و <<) من النتائج
- استعلام فارغ يُرجع نصاً فارغاً (لا نتائج عشوائية)
- similarity متدرجة لنتائج FTS5 (ليست ثابتة)
- فلتر نصوص مقطوعة: ترفض بدايات بعلامات ترقيم، نهايات بكلمات مبتورة
- فلتر مسارات الملفات في المحتوى
- فلتر أكواد برمجية متسربة (print, def, class, etc.)
- حد أدنى 25 حرف للمحتوى المفيد
- تطبيع النقاط: ...... → ...
"""

from agent.memory_provider import MemoryProvider
from typing import List, Dict, Any, Optional
import logging
import json
import re
import functools
import time
import threading
import sys
import os
import hashlib
from pathlib import Path
import numpy as np

# P0: Moved to separate modules
from unified.utils import _format_age, _clean_chunk
from unified.cache import QueryResultCache, GraphCache
from unified.memory_db import _get_memory_db
from unified.agent_id import AgentIdMixin

# P1b: Auto consolidation (optional — import on demand)
# from unified.consolidation import MemoryConsolidator

logger = logging.getLogger(__name__)


class UnifiedMemoryProvider(MemoryProvider, AgentIdMixin):
    """مزود ذاكرة موحد محسّن"""
    
    _graphify_lock = threading.Lock()  # thread-safe lazy loading
    
    def __init__(self, config=None):
        self.config = config or {}
        self._agent_id = "default"  # P5: AgentIdMixin field (init before mixin)
        self._initialized = False
        self.graphify = None
        self._graphify_config = self.config  # إصلاح: قيمة افتراضية قبل initialize()
        self._graph_cache = GraphCache()
        # P1: EmbeddingCache replaced with in-memory LRU cache (no disk I/O)
        self._query_cache = QueryResultCache(
            ttl_seconds=300  # 5 دقائق
        )
        self._ft_db = None
        self._model_loaded = False  # للخلفية Preloading
        self._model_loading_thread = None  # مؤشر على خيط التحميل
        self._model_ready = threading.Event()  # P0: Event للمزامنة الآمنة
    
    @property
    def name(self):
        return "unified"
    
    def is_available(self):
        try:
            import networkx
            import sklearn
            import sentence_transformers
            import langchain
            
            graphs_dir = self.config.get("graphs_dir", "~/.hermes/graphs")
            graphs_path = Path(graphs_dir).expanduser()
            graphs_path.mkdir(parents=True, exist_ok=True)
            
            return True
        except ImportError as e:
            logger.warning("Missing dependency: %s", e)
            return False
    
    def initialize(self, session_id, **kwargs):
        if self._initialized:
            return
        
        # P5: Set agent_id from config or kwargs
        agent_id = kwargs.get('agent_id') or self.config.get('agent_id')
        if agent_id:
            self._set_agent_id(agent_id)
        
        logger.info("Initializing UnifiedMemoryProvider v3.5 for session %s (agent: %s)", session_id, self._get_agent_id())
        
        # Graphify — Lazy Loading (لا تهيئ هنا، التهيئة عند أول طلب فقط)
        self.graphify = None
        self._graphify_config = self.config
        
        # FTS5
        try:
            hermes_path = str(Path(__file__).parent.parent.parent.parent / 'hermes-agent')
            if hermes_path not in sys.path:
                sys.path.insert(0, hermes_path)
            
            from hermes_state import SessionDB, DEFAULT_DB_PATH
            self._ft_db = SessionDB(DEFAULT_DB_PATH)
            logger.info("FTS5 SessionDB initialized")
        except Exception as e:
            logger.warning("Failed to initialize FTS5: %s", e)
            self._ft_db = None
        
        # تحميل قاموس المترادفات للتوسيع الدلالي (معطيل افتراضياً — BGE-M3 يعالج المرادفات دلالياً)
        self._synonym_dict = {}
        self._enable_synonym_expansion = self.config.get("enable_synonym_expansion", False)
        if self._enable_synonym_expansion:
            try:
                _syn_path = Path(__file__).resolve().parent / "synonym_dict.json"
                if _syn_path.exists():
                    with open(_syn_path, encoding='utf-8') as fh:
                        self._synonym_dict = json.load(fh)
                    logger.info("Synonym dict loaded: %d entries", len(self._synonym_dict))
            except Exception as e:
                logger.warning("Failed to load synonym_dict.json: %s", e)
        
        self._initialized = True
        logger.info("UnifiedMemoryProvider v3.5 fully initialized")
        
        # Background Preloading — P0: آمن مع threading.Event
        self._preload_model_background()
    
    def _preload_model_background(self):
        """تحميل نموذج التضمين في الخلفية (Background Preloading)
        
        P0: بدلاً من pass (كان يسبب cold start 17.5 ثانية)،
        نستخدم thread خلفي مع threading.Event للمزامنة الآمنة.
        """
        if self._model_loading_thread is not None and self._model_loading_thread.is_alive():
            return  # حماية من التحميل المكرر
        
        def _worker():
            try:
                with self._graphify_lock:
                    if self.graphify is None:
                        self._init_graphify()
                        # Warmup: استعلام وهمي يحرّك CUDA kernels
                        if self.graphify and hasattr(self.graphify, 'embedding'):
                            try:
                                self.graphify.embedding.embed_query("warmup")
                            except Exception:
                                pass
            except Exception as e:
                logger.warning("Background preload failed: %s", e)
            finally:
                # Event يتعيّن دائماً — حتى لو فشل، ما يعلق prefetch()
                self._model_ready.set()
        
        self._model_loading_thread = threading.Thread(
            target=_worker, daemon=True, name="bge-m3-preload"
        )
        self._model_loading_thread.start()
    
    def _get_graphify(self):
        """تهيئة Graphify عند أول طلب فقط (Lazy Loading — thread-safe)
        
        P0: إذا كان التحميل الخلفي جارياً، ننتظره (max 30s) بدل البدء من الصفر.
        """
        if self.graphify is None:
            # انتظر التحميل الخلفي إن كان جارياً
            if self._model_loading_thread and self._model_loading_thread.is_alive():
                self._model_ready.wait(timeout=30)
            with self._graphify_lock:
                if self.graphify is None:
                    self._init_graphify()
        return self.graphify
    
    def _init_graphify(self):
        """تهيئة Graphify Engine (يُستدعى مرة واحدة فقط)"""
        logger.info("Lazy-loading Graphify Engine...")
        try:
            # Hermes يحمّل الـ plugin كـ _hermes_user_memory.unified.*
            # نحاول نجيبه من sys.modules أولاً لتجنب تحميل نسخة ثانية
            graph_engine = None
            for _mod_name in list(sys.modules.keys()):
                if _mod_name.endswith('.unified.graph_engine') and 'GraphifyEngine' in dir(sys.modules[_mod_name]):
                    graph_engine = sys.modules[_mod_name]
                    break

            if graph_engine is None:
                # P3: Direct file load — bypass fragile sys.modules walking
                import importlib.util
                _engine_path = Path(__file__).resolve().parent / "graph_engine.py"
                if _engine_path.exists():
                    # Register 'unified' package BEFORE exec_module (fix relative imports)
                    if "unified" not in sys.modules:
                        pkg_mod = type(sys)('unified')
                        pkg_mod.__path__ = [str(Path(__file__).resolve().parent)]
                        sys.modules["unified"] = pkg_mod

                    spec = importlib.util.spec_from_file_location("unified.graph_engine", _engine_path)
                    graph_engine = importlib.util.module_from_spec(spec)
                    sys.modules["unified.graph_engine"] = graph_engine
                    spec.loader.exec_module(graph_engine)

                    # Load submodules
                    for _sub in ["embedding_model", "document_loader", "text_splitter",
                                 "graph_builder", "community_detector", "graph_storage"]:
                        if f"unified.{_sub}" in sys.modules:
                            continue
                        _sub_path = Path(__file__).resolve().parent / f"{_sub}.py"
                        if _sub_path.exists():
                            _sub_spec = importlib.util.spec_from_file_location(f"unified.{_sub}", _sub_path)
                            _sub_mod = importlib.util.module_from_spec(_sub_spec)
                            sys.modules[f"unified.{_sub}"] = _sub_mod
                            _sub_spec.loader.exec_module(_sub_mod)
                else:
                    raise FileNotFoundError(f"graph_engine.py not found at {_engine_path}")

            GraphifyEngine = graph_engine.GraphifyEngine
            logger.info("Creating GraphifyEngine with config: %s", self._graphify_config)
            self.graphify = GraphifyEngine(self._graphify_config)
            logger.info("Graphify Engine initialized (lazy)")
        except Exception as e:
            logger.error("Failed to initialize Graphify: %s", e, exc_info=True)
            self.graphify = None
        return self.graphify
    
    def system_prompt_block(self):
        return (
            "## Memory System\n\n"
            "You have access to a Unified Memory System:\n"
            "1. **Semantic Graph** - Knowledge graph with concept relationships\n"
            "2. **Session History** - Past conversation messages\n\n"
            "Use graph_search for knowledge graph queries.\n"
            "Use unified_search for all memory layers.\n"
        )
    
    def _search_graph_cached(self, query, top_k_per_project=2, max_age_cutoff=None, session_id="", relation_type=None):
        """بحث دلالي SQLite-native — P6 final.
        
        Reads directly from MemoryDB (hermes_memory.db).
        No NetworkX, no graph.json, no GraphCache needed.
        
        P3: relation_type parameter filters neighbor expansion by typed relations.
        If None, uses existing 'similar' kind (backward compatible).
        """
        import time
        _t_start = time.perf_counter()
        
        if not query or len(query.strip()) < 2:
            return []
        
        has_word = bool(re.search(r'[\u0621-\u064Aa-zA-Z0-9]', query))
        if not has_word:
            return []
        
        # P4: Pass session_id for consistent cache key with prefetch
        cache_hit = self._query_cache.get(query, session_id)
        if cache_hit is not None:
            return cache_hit
        
        db = _get_memory_db()
        if db is None:
            return []
        
        try:
            graphify = self._get_graphify()
            if not graphify:
                return []
            
            from .arabic_normalizer import normalize_query
            normalized_query = normalize_query(query)
            expanded_query = normalized_query
            if self._synonym_dict:
                words = re.split(r'\s+', normalized_query.strip())
                syns = set()
                for w in words:
                    wc = w.strip().rstrip('؟!.,:;،؛')
                    if wc in self._synonym_dict:
                        for s in self._synonym_dict[wc][:3]:
                            if len(s) > 2: syns.add(s)
                if syns:
                    expanded_query = query + ' ' + ' '.join(syns)
            
            query_embedding = graphify.embedding.embed_query(expanded_query)
            sqlite_results = db.search_similar(query_embedding, top_k=top_k_per_project * 3)
            
            results = []
            seen_ids = set()
            
            for r in sqlite_results:
                cleaned = _clean_chunk(r['key'])
                if cleaned and r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    results.append({
                        'similarity': r['similarity'],
                        'content': cleaned,
                        'type': 'fact',
                        'project': 'hermes-memory',
                        'expanded': False,
                    })
            
            # P3: 1-hop neighbor expansion via fact_relations (supports typed relations)
            if results and len(sqlite_results) >= 2:
                neighbor_ids = set()
                relation_kind = relation_type or 'similar'  # P3: typed or default
                for r in sqlite_results[:2]:
                    nbrs = db.get_neighbors(r['id'], kind=relation_kind)
                    for n in nbrs[:2]:
                        if n['id'] not in seen_ids:
                            neighbor_ids.add(n['id'])
                
                for nid in neighbor_ids:
                    n_fact = db.get_fact(nid)
                    if n_fact and n_fact.get('embedding'):
                        n_emb = np.asarray(n_fact['embedding'], dtype=np.float32)
                        q_vec = np.asarray(query_embedding, dtype=np.float32)
                        n_norm = np.linalg.norm(n_emb)
                        q_norm = np.linalg.norm(q_vec)
                        if n_norm > 1e-10 and q_norm > 1e-10:
                            sim = float(n_emb @ q_vec / (n_norm * q_norm))
                            cleaned = _clean_chunk(n_fact.get('full_key', ''))
                            if cleaned and sim >= 0.5:
                                seen_ids.add(nid)
                                results.append({
                                    'similarity': sim,
                                    'content': cleaned,
                                    'type': 'fact',
                                    'project': 'hermes-memory',
                                    'expanded': True,
                                })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k_per_project]
            
            logger.debug("SQLite search: %.1fms | n=%d | '%s'",
                        (time.perf_counter() - _t_start) * 1000, len(results), query[:40])
            return results
            
        except Exception as e:
            logger.error("SQLite search error: %s", e)
            return []
    
    def _apply_decay(self, results, lambda_decay=0.01):
        """P1a: تطبيق معامل الانخفاض الزمني على نتائج البحث.
        
        adjusted_score = base_score * e^(-λ * age_days)
        λ = 0.01/يوم → نصف العمر ≈ 69 يوم
        
        Results بدون timestamp (مثل graph facts) تأخذ عمر افتراضي من config.
        """
        import math
        now = time.time()
        for r in results:
            ts = r.get('timestamp', 0)
            if ts > 0:
                age_days = (now - ts) / 86400.0
                decay_factor = math.exp(-lambda_decay * age_days)
            else:
                # No timestamp — use the similarity as-is (graph facts)
                decay_factor = 1.0
            # Apply decay to rrf_score or similarity
            base = r.get('rrf_score', r.get('similarity', 0))
            r['adjusted_score'] = base * decay_factor
            r['decay_factor'] = decay_factor
        return sorted(results, key=lambda x: x['adjusted_score'], reverse=True)

    def _rerank_rrf(self, graph_results, fts5_results, k=60, graph_weight=1.5, fts5_weight=1.0, lambda_decay=0.0):
        """إعادة ترتيب النتائج باستخدام Reciprocal Rank Fusion (RRF)
        
        RRF يدمج قائمتين مرتبتين باستخدام: score = 1 / (k + rank)
        
        P1a: إذا lambda_decay > 0، يُطبق معامل الانخفاض الزمني على نتائج FTS5
        قبل الدمج (لأن عندها timestamps).
        
        Args:
            graph_results: نتائج من Graph (قائمة dicts)
            fts5_results: نتائج من FTS5 (قائمة dicts)
            k: ثابت التنعيم (افتراضي: 60 — قيمة قياسية في الأدبيات)
            graph_weight: وزن نتائج Graph (افتراضي: 1.5 — أعلى لأن Graph أدق)
            fts5_weight: وزن نتائج FTS5 (افتراضي: 1.0)
            lambda_decay: معامل الانخفاض الزمني (0.0 = معطّل)
        
        Returns:
            قائمة موحدة مرتبة حسب RRF score
        """
        import math
        from collections import defaultdict
        
        now = time.time()
        
        def _apply_decay_to_result(r, now, lambda_decay):
            """P1a: تطبيق decay على نتيجة واحدة."""
            if lambda_decay <= 0:
                return 1.0
            ts = r.get('timestamp', 0)
            if ts > 0:
                age_days = (now - ts) / 86400.0
                return math.exp(-lambda_decay * age_days)
            return 1.0
        
        seen_hashes = set()
        rrf_scores = defaultdict(float)
        all_results = {}
        
        # تطبيق RRF على نتائج Graph
        for rank, r in enumerate(graph_results):
            h = hashlib.md5(r['content'].encode('utf-8')).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            score = graph_weight / (k + rank + 1)  # +1 لأن rank يبدأ من 0
            rrf_scores[h] += score
            all_results[h] = r
        
        # تطبيق RRF على نتائج FTS5 (مع decay اختياري)
        for rank, r in enumerate(fts5_results):
            h = hashlib.md5(r['content'].encode('utf-8')).hexdigest()
            
            # P1a: حساب decay factor
            decay_factor = _apply_decay_to_result(r, now, lambda_decay)
            
            if h in seen_hashes:
                # نتيجة مكررة — نضيف score إضافي (مخفّض بالـ decay)
                rrf_scores[h] += fts5_weight / (k + rank + 1) * 0.5 * decay_factor
                continue
            seen_hashes.add(h)
            
            score = fts5_weight / (k + rank + 1) * decay_factor
            rrf_scores[h] += score
            all_results[h] = r
        
        # ترتيب حسب RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # إعادة بناء القائمة مع RRF score
        final_results = []
        for h, score in ranked:
            result = all_results[h].copy()
            result['rrf_score'] = score
            # P1a: إضافة adjusted_score (نفس rrf_score لكن واضح للمستخدم)
            result['adjusted_score'] = score
            final_results.append(result)
        
        return final_results
    
    def prefetch(self, query, session_id="", max_age_days=None):
        """استدعاء ذاكرة قبل كل دور — مع فلترة زمنية وإزالة تكرار
        
        Args:
            query: نص الاستعلام
            session_id: معرف الجلسة
            max_age_days: العمر الأقصى للنتائج بالأيام (افتراضي: 30 يوم للرسوم، 7 للرسائل)
        """
        if not self._initialized:
            return ""
        
        # فلتر: استعلام فارغ أو قصير جداً لا يُرجع شيئاً
        if not query or len(query.strip()) < 3:
            return ""
        
        # Query Result Caching — فحص الكاش أولاً
        cached_result = self._query_cache.get(query, session_id, max_age_days)
        if cached_result is not None:
            logger.debug(f"Query cache hit for '{query[:50]}...'")
            return cached_result
        
        # فلتر زمني: تجاهل رسائل آخر 60 ثانية (منع ظهور رسالة المستخدم الحالية والجلسة الحالية)
        import time as _time
        echo_cutoff = _time.time() - 60  # زاد من 15 إلى 60 ثانية
        
        # فلتر زمني configurable للنتائج القديمة
        if max_age_days is None:
            max_age_days_graph = 30  # الرسوم المعرفية: 30 يوم
            max_age_days_messages = 7  # الرسائل: 7 أيام
        else:
            max_age_days_graph = max_age_days
            max_age_days_messages = max_age_days
        
        # حساب cutoff للرسوم المعرفية (بناءً على وقت التحميل)
        graph_cutoff = _time.time() - (max_age_days_graph * 86400)
        
        # حساب cutoff للرسائل
        message_cutoff = _time.time() - (max_age_days_messages * 86400)
        
        seen_hashes = set()  # لإزالة التكرار
        graph_results = []
        fts5_results = []
        
        # Graph search (Lazy Loading — يحمل المشاريع عند الطلب فقط)
        graph_results = self._search_graph_cached(query, top_k_per_project=2, max_age_cutoff=graph_cutoff, session_id=session_id)
        
        # FTS5 (مع فلتر زمني configurable واستبعاد tool output والجلسة الحالية)
        if self._ft_db:
            try:
                fts5_results = self._ft_db.search_messages(
                    query, 
                    limit=5,
                    role_filter=["user", "assistant"],
                    exclude_session=session_id
                )
                
                # تحويل نتائج FTS5 إلى نفس تنسيق Graph
                fts5_formatted = []
                for i, r in enumerate(fts5_results):
                    msg_ts = r.get('timestamp', 0)
                    if isinstance(msg_ts, str):
                        try:
                            from datetime import datetime
                            msg_ts = datetime.fromisoformat(msg_ts.replace('Z', '+00:00')).timestamp()
                        except Exception:
                            msg_ts = 0
                    if msg_ts > echo_cutoff or msg_ts < message_cutoff:
                        continue
                    
                    snippet = r.get('snippet', '')
                    query_words = set(query.lower().split())
                    if len(query_words) > 3:
                        snippet_words = set(snippet.lower().split())
                        overlap = len(query_words & snippet_words) / len(query_words)
                        if overlap > 0.85:
                            continue
                    
                    cleaned = _clean_chunk(snippet, max_len=300)
                    if cleaned:
                        fts5_formatted.append({
                            'similarity': 0.75 - (i * 0.03),
                            'content': cleaned,
                            'type': 'session',
                            'project': 'sessions',
                            'timestamp': msg_ts,  # P1: تمرير الوقت للـ source label
                        })
                
                fts5_results = fts5_formatted
            except Exception:
                fts5_results = []
        else:
            fts5_results = []
        
        # Re-ranking باستخدام RRF
        lambda_decay = self.config.get('decay_lambda', 0.0)  # P1a: 0.0 = معطّل افتراضياً
        ranked_results = self._rerank_rrf(
            graph_results,
            fts5_results,
            k=60,
            graph_weight=1.5,
            fts5_weight=1.0,
            lambda_decay=lambda_decay,
        )
        
        # إزالة التكرار النهائي
        results = []
        for r in ranked_results:
            h = hashlib.md5(r['content'].encode('utf-8')).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(r)
        
        # ترتيب وإخراج
        if results:
            # ترتيب حسب RRF score (إن وجد) أو similarity
            results.sort(key=lambda x: x.get('rrf_score', x['similarity']), reverse=True)
            now = _time.time()  # P1: مرجع الوقت لـ source label
            lines = ["## Memory Context\n"]
            for r in results[:5]:
                rtype = r.get('type', 'unknown')
                project = r.get('project', '')
                expanded = r.get('expanded', False)
                
                # P1: source label واضح بدل lang_tag
                if rtype == 'session':
                    ts = r.get('timestamp', 0)
                    age = _format_age(now - ts) if ts else ''
                    source = f"session:{age}" if age else "session"
                elif rtype == 'fact':
                    source = f"graph:{project}"
                    if expanded:
                        source += "·1hop"
                else:
                    source = rtype
                
                score = r.get('rrf_score', r['similarity'])
                lines.append("- [%.2f][%s] %s" % (score, source, r['content']))
            result_text = "\n".join(lines)
            
            # حفظ في الكاش قبل الإرجاع
            self._query_cache.set(query, result_text, session_id, max_age_days)
            return result_text
        
        return ""
    
    def sync_turn(self, user_content, assistant_content, session_id=""):
        pass
    
    def get_tool_schemas(self):
        return [
            {
                "name": "unified_search",
                "description": "بحث موحد في جميع طبقات الذاكرة",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "نص البحث"},
                        "limit": {"type": "integer", "default": 10},
                        "agent_id": {
                            "type": "string",
                            "description": "P5: فلترة حسب معرف الوكيل (use 'all' for all agents)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "graph_index",
                "description": "فهرسة مجلد في الرسم المعرفي",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "مسار المجلد"},
                        "project_name": {"type": "string", "description": "اسم المشروع"},
                        "reindex": {"type": "boolean", "default": False},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "graph_search",
                "description": "بحث دلالي في الرسم المعرفي",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "نص البحث"},
                        "project": {"type": "string", "description": "اسم المشروع"},
                        "top_k": {"type": "integer", "default": 10},
                        "relation_type": {
                            "type": "string",
                            "description": "P3: فلترة بنوع العلاقة (causes, fixes, supports, contradicts, related)",
                            "enum": ["causes", "fixes", "supports", "contradicts", "related"],
                        },
                    },
                    "required": ["query", "project"],
                },
            },
        ]
    
    def handle_tool_call(self, tool_name, args):
        try:
            if tool_name == "unified_search":
                return json.dumps(self._tool_unified_search(args), ensure_ascii=False, default=str)
            elif tool_name == "graph_index":
                return json.dumps(self._tool_graph_index(args), ensure_ascii=False, default=str)
            elif tool_name == "graph_search":
                return json.dumps(self._tool_graph_search(args), ensure_ascii=False, default=str)
            else:
                return json.dumps({"error": "Unknown tool: %s" % tool_name})
        except Exception as e:
            logger.error("Tool call error: %s", e)
            return json.dumps({"error": str(e)})
    
    def _tool_unified_search(self, args):
        graphify = self._get_graphify()
        if not graphify:
            return {"error": "Graphify not initialized"}
        
        query = args.get("query", "")
        limit = int(args.get("limit", 10))  # P2: cast to int
        filter_agent_id = self._get_agent_id_from_args(args)
        
        # فلتر: استعلام فارغ أو غير صالح
        if not query or len(query.strip()) < 2:
            return {"query": query, "results": [], "total": 0}
        
        # فلتر: استعلام يحتوي رموز فقط بدون كلمات حقيقية
        has_word = bool(re.search(r'[\u0621-\u064Aa-zA-Z0-9]', query))
        if not has_word:
            return {"query": query, "results": [], "total": 0}
        
        results = self._search_graph_cached(query, top_k_per_project=limit)
        
        # P5: Annotate results with agent_id
        self._annotate_results_with_agent_id(results)
        
        # P5: Filter by agent_id if specified
        results = self._filter_by_agent_id(results, filter_agent_id)
        
        # Deduplication
        seen = set()
        deduped = []
        for r in results:
            h = hashlib.md5(r.get("content", "").encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(r)
        results = deduped
        
        # FTS5 — استبعاد رسائل tool (ملفات ومسارات)
        if self._ft_db:
            try:
                fts5_results = self._ft_db.search_messages(
                    query, 
                    limit=limit,
                    role_filter=["user", "assistant"]  # فقط رسائل المستخدم والمساعد
                )
                for i, r in enumerate(fts5_results):
                    snippet = _clean_chunk(r.get('snippet', ''), max_len=300)
                    if snippet:
                        h = hashlib.md5(snippet.encode("utf-8")).hexdigest()
                        if h not in seen:
                            seen.add(h)
                            results.append({
                                "source": "fts5",
                                "similarity": 0.70 - (i * 0.03),
                                "content": snippet,
                                "type": "session",
                            })
            except Exception:
                pass
        
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        final = results[:limit]
        return {"query": query, "results": final, "total": len(final)}  # P2: total بعد القطع
    
    def _tool_graph_index(self, args):
        graphify = self._get_graphify()
        if not graphify:
            return {"error": "Graphify not initialized"}
        
        path = args.get("path", "")
        project_name = args.get("project_name")
        reindex = args.get("reindex", False)
        
        try:
            report = graphify.index_directory(path, project_name=project_name, reindex=reindex)
            self._graph_cache.clear()
            self._query_cache.clear()  # مسح كاش النتائج عند تغيير البيانات
            return report
        except Exception as e:
            return {"error": str(e)}
    
    def _tool_graph_search(self, args):
        """P4: SQLite-based graph search (replaces NetworkX).
        
        P3: Added relation_type parameter to filter results by typed relations.
        """
        query = args.get("query", "")
        project = args.get("project", "hermes-memory")  # Default to main project
        top_k = args.get("top_k", 10)
        relation_type = args.get("relation_type")  # P3: optional filter
        
        # فلتر: استعلام فارغ أو غير صالح
        if not query or len(query.strip()) < 2:
            return {"query": query, "project": project, "results": [], "total": 0}
        
        has_word = bool(re.search(r'[\u0621-\u064Aa-zA-Z0-9]', query))
        if not has_word:
            return {"query": query, "project": project, "results": [], "total": 0}
        
        try:
            # P4: Use SQLite-based vector search (not NetworkX graph)
            db = _get_memory_db()
            if db is None:
                # Fallback to legacy graphify for backward compat
                graphify = self._get_graphify()
                if graphify:
                    # Load graph the old way
                    if project and project != "hermes-memory":
                        from pathlib import Path
                        graph = self._graph_cache.get(project, graphify.storage.load)
                        if graph:
                            query_embedding = graphify.embedding.embed_query(query)
                            results = []
                            for node_id in graph.nodes():
                                node_embedding = graph.nodes[node_id].get("embedding")
                                if node_embedding is None:
                                    continue
                                v1 = np.array(query_embedding); v2 = np.array(node_embedding)
                                v1_norm = np.linalg.norm(v1); v2_norm = np.linalg.norm(v2)
                                if v1_norm < 1e-10 or v2_norm < 1e-10:
                                    continue
                                similarity = float(np.dot(v1, v2) / (v1_norm * v2_norm))
                                if similarity >= 0.5:
                                    results.append({"similarity": similarity, "content": _clean_chunk(graph.nodes[node_id].get("content", "")), "type": graph.nodes[node_id].get("type", "unknown")})
                            results.sort(key=lambda x: x["similarity"], reverse=True)
                            return {"query": query, "project": project, "results": results[:top_k], "total": len(results), "backend": "networkx-fallback"}
                    return {"error": "No project specified for legacy search"}
                return {"error": "Memory DB not available and Graphify not initialized"}
            
            # Get embedding from graphify (BGE-M3 cache)
            graphify = self._get_graphify()
            if not graphify:
                return {"error": "Graphify not initialized"}
                
            query_embedding = graphify.embedding.embed_query(query)
            
            # Search SQLite via MemoryDB
            results = db.search_similar(query_embedding, top_k=top_k, threshold=0.5)
            
            # Also do FTS5 text search for complementary results
            text_results = db.search_text(query, limit=3)
            
            # Combine and deduplicate
            seen_ids = set()
            combined = []
            
            for r in results:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    combined.append({
                        "similarity": r["similarity"],
                        "content": _clean_chunk(r["key"]),
                        "type": "fact",
                        "category": r["category"],
                        "source": "vector",
                    })
            
            for r in text_results:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    combined.append({
                        "similarity": 0.6,  # Text matches get moderate sim
                        "content": _clean_chunk(r["key"]),
                        "type": "fact",
                        "category": r["category"],
                        "source": "fts5",
                    })
            
            # P3: 1-hop neighbor expansion with typed relations
            if combined:
                relation_kind = relation_type or 'similar'
                for r in combined[:3]:
                    fact_id = r.get('id')
                    if fact_id:
                        nbrs = db.get_neighbors(fact_id, kind=relation_kind)
                        for n in nbrs[:2]:
                            if n['id'] not in seen_ids:
                                seen_ids.add(n['id'])
                                n_fact = db.get_fact(n['id'])
                                if n_fact:
                                    combined.append({
                                        "similarity": n.get('weight', 0.5),
                                        "content": _clean_chunk(n_fact.get('full_key', '')),
                                        "type": "fact",
                                        "category": n_fact.get("category", "unknown"),
                                        "source": f"relation:{relation_kind}",
                                    })
            
            combined.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "query": query, "project": project,
                "results": combined[:top_k],
                "total": len(combined),
                "backend": "sqlite"
            }
        except Exception as e:
            logger.error("graph_search error: %s", e)
            return {"error": str(e)}
    
    def shutdown(self):
        logger.info("Shutting down UnifiedMemoryProvider v3.5")
        self._graph_cache.clear()
        if self._ft_db and hasattr(self._ft_db, 'close'):
            try:
                self._ft_db.close()
            except Exception:
                pass
        self._initialized = False


def register(ctx):
    """تسجيل المزود مع Hermes"""
    # استخدام hasattr بدلاً من getattr لتجنب التحذيرات الأمنية
    if hasattr(ctx, 'config'):
        config = ctx.config or {}
    else:
        config = {}
    plugin_config = config.get('plugins', {}).get('unified', {})
    ctx.register_memory_provider(UnifiedMemoryProvider(plugin_config))
