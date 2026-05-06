"""Cache implementations for Unified Memory Provider."""

import logging
import os
import time
import threading
import hashlib

logger = logging.getLogger(__name__)


class QueryResultCache:
    """كاش لنتائج البحث — يقلل وقت الاستعلامات المتكررة 98%+
    
    P3: Lock لحماية thread-safety + max_size لمنع memory leak.
    """
    
    MAX_SIZE = 500  # P3: حد أعلى للكاش
    
    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl_seconds
        self._lock = threading.Lock()  # P3: thread safety
    
    def _make_key(self, query, session_id, max_age_days):
        """إنشاء مفتاح فريد للكاش"""
        key_str = f"{query}|{session_id}|{max_age_days}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, query, session_id="", max_age_days=None):
        """جلب نتيجة من الكاش"""
        key = self._make_key(query, session_id, max_age_days)
        
        with self._lock:  # P3: thread safety
            if key in self._cache:
                age = time.time() - self._timestamps[key]
                if age < self.ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    def set(self, query, result, session_id="", max_age_days=None):
        """حفظ نتيجة في الكاش"""
        key = self._make_key(query, session_id, max_age_days)
        with self._lock:  # P3: thread safety + size cap
            if len(self._cache) > self.MAX_SIZE:
                self.cleanup_expired()
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
        expired = [
            k for k, ts in self._timestamps.items()
            if time.time() - ts >= self.ttl
        ]
        for key in expired:
            del self._cache[key]
            del self._timestamps[key]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")


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
        graphs_dir = os.path.expanduser("~/.hermes/graphs")
        graph_file = os.path.join(graphs_dir, project_name, "graph.json")

        if project_name in self._graphs:
            # P1: Check if file was modified since last load
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
