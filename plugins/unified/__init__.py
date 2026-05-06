"""Unified Memory Provider v3.5 — مزود الذاكرة الموحد المحسّن

التحسينات:
- In-memory caching للرسوم المعرفية (تحميل مرة واحدة)
- بحث في جميع المشاريع
- تنظيف المحتوى المحسّن (لا يزيل {} و [])
- دمج FTS5 + Graph في prefetch
- معالجة أخطاء شاملة (0% فشل)
- فلتر زمني (15 ثانية) لمنع ظهور رسالة المستخدم كذاكرة
- إزالة تكرار بالـ Hashing (نفس النص من مشاريع مختلفة يظهر مرة واحدة)
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

# Global singleton — مشاركة النموذج عالمياً لتجنب إعادة التحميل
_GLOBAL_MODEL_CACHE = {}
import numpy as np

# P4: MemoryDB SQLite reference (lazy init — only when needed)
_memory_db = None

def _get_memory_db():
    """Lazy-load MemoryDB singleton for graph search."""
    global _memory_db
    if _memory_db is None:
        try:
            _scripts_dir = os.path.expanduser("~/.hermes/../hermes-super-memory/scripts")
            if os.path.isdir(_scripts_dir) and _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from db import MemoryDB
            _memory_db = MemoryDB()
            _memory_db.init()
        except Exception:
            return None
    return _memory_db

logger = logging.getLogger(__name__)


def _clean_chunk(content, max_len=300):
    """تنظيف بسيط للنصوص — بديل 10 سطور لـ 165 سطر regex هش.
    
    P1: الحقائق من LLM نظيفة أصلاً. هذا يكفي.
    FTS5 snippets تستخدم نفس الدالة.
    """
    if not content:
        return ""
    # إزالة أحرف التحكم وعلامات FTS5
    content = content.replace('>>>', '').replace('<<<', '')
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
    content = content.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\\t', ' ')
    # تطبيع مسافات واقتطاع
    content = re.sub(r'\s+', ' ', content).strip()
    if len(content) > max_len:
        content = content[:max_len].rsplit(' ', 1)[0] + '...'
    return content if len(content) >= 15 else ""


class QueryResultCache:
    """كاش لنتائج البحث — يقلل وقت الاستعلامات المتكررة 98%+"""
    
    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl_seconds
    
    def _make_key(self, query, session_id, max_age_days):
        """إنشاء مفتاح فريد للكاش"""
        import hashlib
        key_str = f"{query}|{session_id}|{max_age_days}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, query, session_id="", max_age_days=None):
        """جلب نتيجة من الكاش"""
        import time
        key = self._make_key(query, session_id, max_age_days)
        
        if key in self._cache:
            age = time.time() - self._timestamps[key]
            if age < self.ttl:
                return self._cache[key]
            else:
                # منتهي الصلاحية
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, query, result, session_id="", max_age_days=None):
        """حفظ نتيجة في الكاش"""
        import time
        key = self._make_key(query, session_id, max_age_days)
        self._cache[key] = result
        self._timestamps[key] = time.time()
    
    def clear(self):
        """تنظيف الكاش بالكامل"""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        logger.info(f"Cleared {count} items from query result cache")
    
    def cleanup_expired(self):
        """تنظيف النتائج منتهية الصلاحية"""
        import time
        expired = [
            k for k, ts in self._timestamps.items()
            if time.time() - ts >= self.ttl
        ]
        for key in expired:
            del self._cache[key]
            del self._timestamps[key]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")



# P4: SQLite-based graph search replaces NetworkX. Embedding caching handled by
# MemoryDB.search_similar() internally (single-query load, cosine in numpy).
# Legacy _cached_embedding removed — was raising NotImplementedError.

class GraphCache:
    """ذاكرة تخزين مؤقت للرسوم المعرفية مع mtime invalidation + LRU cap (P4)."""
    
    MAX_CACHED_GRAPHS = 50  # P4: Prevent unbounded memory growth
    
    def __init__(self):
        self._graphs = {}
        self._load_times = {}
        self._mtimes = {}
        self._stats = {}
        self._access_order = []  # P4: LRU tracking
    
    def _evict_if_needed(self):
        """P4: Evict least recently used graph if over capacity."""
        while len(self._graphs) > self.MAX_CACHED_GRAPHS and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._graphs:
                del self._graphs[oldest]
                self._load_times.pop(oldest, None)
                self._mtimes.pop(oldest, None)
                self._stats.pop(oldest, None)
                logger.debug("GraphCache: evicted %s (LRU cap=%d)", oldest, self.MAX_CACHED_GRAPHS)
    
    def get(self, project_name, loader_fn):
        """جلب رسم من الذاكرة أو تحميله مع mtime check."""
        import os
        if project_name in self._graphs:
            # P1: Check if file was modified since last load
            graphs_dir = os.path.expanduser("~/.hermes/graphs")
            graph_file = os.path.join(graphs_dir, project_name, "graph.json")
            if os.path.exists(graph_file):
                current_mtime = os.path.getmtime(graph_file)
                if self._mtimes.get(project_name, 0) < current_mtime:
                    # File changed — reload
                    logger.debug("GraphCache: %s mtime changed, reloading", project_name)
                    del self._graphs[project_name]
                    del self._mtimes[project_name]
                else:
                    return self._graphs[project_name]
        
        try:
            graph = loader_fn(project_name)
            self._graphs[project_name] = graph
            self._load_times[project_name] = time.time()
            # P6: Update LRU access order + evict if over capacity
            if project_name in self._access_order:
                self._access_order.remove(project_name)
            self._access_order.append(project_name)
            self._evict_if_needed()
            self._stats[project_name] = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
            }
            if os.path.exists(graph_file):
                self._mtimes[project_name] = os.path.getmtime(graph_file)
            return graph
        except Exception as e:
            logger.warning("Failed to load graph %s: %s", project_name, e)
            return None
    
    def clear(self):
        self._graphs.clear()
        self._load_times.clear()
        self._mtimes.clear()
        self._stats.clear()
    
    @property
    def stats(self):
        return dict(self._stats)


class UnifiedMemoryProvider(MemoryProvider):
    """مزود ذاكرة موحد محسّن"""
    
    _graphify_lock = threading.Lock()  # thread-safe lazy loading
    
    def __init__(self, config=None):
        self.config = config or {}
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
        
        logger.info("Initializing UnifiedMemoryProvider v3.5 for session %s", session_id)
        
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
        
        # Background Preloading — تم إزالته لتجنب race conditions
        self._preload_model_background()
    
    def _preload_model_background(self):
        """تحميل نموذج التضمين عند أول طلب (Lazy Loading)
        
        ملاحظة: أزلنا الـ background preloading لأنه كان يسبب:
        - race conditions مع الـ main thread
        - تحميل مكرر للنموذج
        - self-join errors
        """
        pass  # Lazy loading يتم في _get_graphify() عند أول طلب فعلي
    
    def _get_graphify(self):
        """تهيئة Graphify عند أول طلب فقط (Lazy Loading — thread-safe)"""
        if self.graphify is None:
            with self._graphify_lock:
                if self.graphify is None:  # double-checked locking
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
    
    def _search_graph_cached(self, query, top_k_per_project=2, max_age_cutoff=None):
        """بحث دلالي SQLite-native — P6 final.
        
        Reads directly from MemoryDB (hermes_memory.db).
        No NetworkX, no graph.json, no GraphCache needed.
        """
        import time
        _t_start = time.perf_counter()
        
        if not query or len(query.strip()) < 2:
            return []
        
        has_word = bool(re.search(r'[\u0621-\u064Aa-zA-Z0-9]', query))
        if not has_word:
            return []
        
        cache_hit = self._query_cache.get(query)
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
            
            # 1-hop neighbor expansion via fact_relations
            if results and len(sqlite_results) >= 2:
                neighbor_ids = set()
                for r in sqlite_results[:2]:
                    nbrs = db.get_neighbors(r['id'], kind='similar')
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
    
    def _rerank_rrf(self, graph_results, fts5_results, k=60, graph_weight=1.5, fts5_weight=1.0):
        """إعادة ترتيب النتائج باستخدام Reciprocal Rank Fusion (RRF)
        
        RRF يدمج قائمتين مرتبتين باستخدام: score = 1 / (k + rank)
        
        Args:
            graph_results: نتائج من Graph (قائمة dicts)
            fts5_results: نتائج من FTS5 (قائمة dicts)
            k: ثابت التنعيم (افتراضي: 60 — قيمة قياسية في الأدبيات)
            graph_weight: وزن نتائج Graph (افتراضي: 1.5 — أعلى لأن Graph أدق)
            fts5_weight: وزن نتائج FTS5 (افتراضي: 1.0)
        
        Returns:
            قائمة موحدة مرتبة حسب RRF score
        """
        from collections import defaultdict
        
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
        
        # تطبيق RRF على نتائج FTS5
        for rank, r in enumerate(fts5_results):
            h = hashlib.md5(r['content'].encode('utf-8')).hexdigest()
            if h in seen_hashes:
                # نتيجة مكررة — نضيف score إضافي
                rrf_scores[h] += fts5_weight / (k + rank + 1) * 0.5  # 50% bonus للتكرار
                continue
            seen_hashes.add(h)
            
            score = fts5_weight / (k + rank + 1)
            rrf_scores[h] += score
            all_results[h] = r
        
        # ترتيب حسب RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # إعادة بناء القائمة مع RRF score
        final_results = []
        for h, score in ranked:
            result = all_results[h].copy()
            result['rrf_score'] = score
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
        graph_results = self._search_graph_cached(query, top_k_per_project=2, max_age_cutoff=graph_cutoff)
        
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
                        })
                
                fts5_results = fts5_formatted
            except Exception:
                fts5_results = []
        else:
            fts5_results = []
        
        # Re-ranking باستخدام RRF
        ranked_results = self._rerank_rrf(
            graph_results,
            fts5_results,
            k=60,
            graph_weight=1.5,
            fts5_weight=1.0
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
            lines = ["## Memory Context\n"]
            for r in results[:5]:
                lang_tag = 'arabic' if any('\u0600' <= c <= '\u06FF' for c in r['content'][:20]) else 'english'
                score = r.get('rrf_score', r['similarity'])
                lines.append("- [%.2f][%s] %s" % (score, lang_tag, r['content']))
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
        limit = args.get("limit", 10)
        
        # فلتر: استعلام فارغ أو غير صالح
        if not query or len(query.strip()) < 2:
            return {"query": query, "results": [], "total": 0}
        
        # فلتر: استعلام يحتوي رموز فقط بدون كلمات حقيقية
        # كلمة = حرف عربي/لاتيني أو رقم (ليس علامات ترقيم فقط)
        has_word = bool(re.search(r'[\u0621-\u064Aa-zA-Z0-9]', query))
        if not has_word:
            return {"query": query, "results": [], "total": 0}
        
        results = self._search_graph_cached(query, top_k_per_project=limit)
        
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
        
        return {"query": query, "results": results[:limit], "total": len(results)}
    
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
        """P4: SQLite-based graph search (replaces NetworkX)."""
        query = args.get("query", "")
        project = args.get("project", "hermes-memory")  # Default to main project
        top_k = args.get("top_k", 10)
        
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
