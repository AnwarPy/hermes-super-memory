# 🧠 Hermes Super Memory — Unified Memory Plugin

**A high-performance, 100% local unified memory system for Hermes Agent**  
نظام ذاكرة موحد عالي الأداء لـ Hermes Agent — 100% محلي، صفر اتصالات خارجية

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Arabic RTL](https://img.shields.io/badge/Language-Arabic%20%7C%20English-brightgreen.svg)]()

---

## 🌟 Overview / نظرة عامة

**EN:** Hermes Super Memory is a production-grade memory plugin that unifies full-text search (FTS5), semantic vector search (BGE-M3), and knowledge graph traversal into a single `prefetch()` call. It provides contextual memory to Hermes Agent before every turn — fully local, zero cloud dependencies.

**AR:** هرمز سوبر ميموري هو نظام ذاكرة إنتاجي يدمج البحث النصي (FTS5)، البحث الدلالي (BGE-M3)، واستكشاف الرسم المعرفي في استدعاء `prefetch()` واحد. يوفر سياق ذاكرة لـ هرمز قبل كل دور — محلي بالكامل، صفر اعتماديات سحابية.

---

## ✨ Features / المميزات

| English | العربية |
|---------|---------|
| 🔍 **Unified Search** — Hybrid FTS5 + BGE-M3 semantic search with RRF re-ranking | 🔍 **بحث موحد** — بحث هجين (FTS5 + BGE-M3) مع إعادة ترتيب RRF |
| ⚡ **71ms average response** — Background model preload eliminates cold start | ⚡ **متوسط استجابة 71ms** — تحميل مسبق للنموذج يزيل التأخير الأول |
| 🇸🇦 **Full Arabic support** — Comprehensive normalization (hamza, yaa, diacritics) | 🇸🇦 **دعم عربي كامل** — تطبيع شامل (همزات، ياء، حركات) |
| 🕸️ **Knowledge Graph** — BGE-M3 embeddings + 1-hop neighbor expansion | 🕸️ **رسم معرفي** — تضمينات BGE-M3 + توسيع جار خطوة واحدة |
| 🔒 **100% local & private** — No data ever leaves your machine | 🔒 **خصوصية كاملة** — لا تغادر بياناتك جهازك |
| 🧵 **Thread-safe** — Producer/consumer pattern with `threading.Event` synchronization | 🧵 **آمن متعدد الخيوط** — نمط المنتج/المستهلك مع مزامنة Event |
| 📦 **Production-hardened** — 100% recall, edge-case handling, query caching (TTL 5m) | 📦 **جاهز للإنتاج** — recall 100%، معالجة حالات الحافة، كاش استعلامات |
| 🏷️ **Source labels** — Clear provenance: `[graph:project]` / `[session:2h ago]` | 🏷️ **ملصقات المصدر** — مصدر واضح: `[graph:مشروع]` / `[session:منذ 2س]` |

---

## 📦 Requirements / المتطلبات

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 4 GB | 8 GB+ (for BGE-M3 on GPU) |
| **GPU** | Not required | NVIDIA RTX 3060+ (CUDA) |
| **Storage** | ~3 GB | ~5 GB (models + graphs) |
| **Hermes Agent** | v1.0+ | Latest from main |

> **Note:** BGE-M3 downloads ~2.3 GB on first run. After that, everything works fully offline.  
> **ملاحظة:** BGE-M3 ينزّل ~2.3 جيجابايت أول مرة. بعد ذلك، كل شيء يعمل بدون إنترنت.

---

## 🚀 Installation / التثبيت

```bash
# 1. Clone the repository / استنساخ المشروع
git clone https://github.com/AnwarPy/hermes-super-memory.git
cd hermes-super-memory

# 2. Run the installer / تشغيل المثبت
chmod +x install.sh && ./install.sh

# 3. Run tests (optional) / اختبار (اختياري)
./test.sh

# 4. Restart Hermes / إعادة تشغيل هرمز
hermes gateway restart
```

---

## 📁 Project Structure / هيكل المشروع

```
hermes-super-memory/
├── install.sh                    # One-click installer / مثبت بنقرة واحدة
├── uninstall.sh                  # Clean removal / إزالة نظيفة
├── test.sh                       # Test runner / مشغل الاختبارات
├── README.md                     # This file / هذا الملف
│
├── plugins/unified/              # Core plugin / القلب الرئيسي
│   ├── __init__.py               # UnifiedMemoryProvider — prefetch, RRF, tools
│   ├── embedding_model.py        # BGE-M3 singleton on CUDA/CPU
│   ├── graph_engine.py           # GraphifyEngine — indexing + search
│   ├── graph_builder.py          # Builds knowledge graphs with Arabic normalization
│   ├── graph_storage.py          # Compact JSON graph storage
│   ├── community_detector.py     # Leiden/Louvain community detection
│   ├── document_loader.py        # File/document loader with filters
│   ├── text_splitter.py          # Arabic-aware text chunking (512/96)
│   ├── arabic_normalizer.py      # Comprehensive Arabic text normalization
│   ├── plugin.yaml               # Plugin configuration / إعدادات الإضافة
│   └── tests/                    # 179 tests / 179 اختبار
│       ├── test_arabic_normalizer.py    # 43 tests — Arabic normalization
│       ├── test_document_loader.py      # 34 tests — Document loading
│       ├── test_unified_memory.py       # 33 tests — Unified search
│       ├── test_graph_builder.py        # 29 tests — Graph building
│       ├── test_graph_storage.py        # 25 tests — Graph storage
│       └── test_community_detector.py   # 15 tests — Community detection
│
└── scripts/                      # Cron scripts / سكربتات الجدولة
    ├── session_summarizer.py     # Auto-summarize sessions / تلخيص تلقائي
    ├── fact_extractor.py         # Extract classified facts / استخراج حقائق
    └── graph_updater.py          # Update knowledge graph / تحديث الرسم
```

---

## 🔍 Search Architecture / هندسة البحث

```
┌─────────────────────────────────────────────────────────┐
│              prefetch() — Unified Memory                 │
│                                                         │
│  Query → QueryResultCache(TTL 5m) → MISS                │
│                                                         │
│  ┌──────────────┐    ┌───────────────────────────┐      │
│  │  FTS5        │    │  BGE-M3 Semantic Search   │      │
│  │  (sessions)  │    │  (hermes_memory.db)       │      │
│  │              │    │  ↓ 1-hop expansion        │      │
│  └──────┬───────┘    └──────────┬────────────────┘      │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌───────────────────────────────────────────┐          │
│  │     RRF Re-ranking                        │          │
│  │     score = 1/(60 + rank)                 │          │
│  │     graph_weight=1.5  ×  fts5_weight=1.0  │          │
│  └───────────────────────────────────────────┘          │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────────┐                       │
│  │  ## Memory Context           │                       │
│  │  - [0.87][graph:hermes-memory] ...                  │
│  │  - [0.74][session:2h ago] ...                       │
│  │  - [0.68][graph:hermes-memory·1hop] ...              │
│  └──────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘

Background Pipeline (Cron):
┌─────────────────────────────────────────────────────┐
│ [Every 60m]  Session Summarizer (Ollama local)      │
│   → summaries/*.json                                 │
│   → facts_auto/*.jsonl                               │
│                                                      │
│ [Every 360m] Fact Extractor + Graph Updater (BGE-M3) │
│   → hermes-memory graph with 1024-dim embeddings     │
│   → Auto-discoverable by unified_search              │
└─────────────────────────────────────────────────────┘
```

---

## ⚡ Performance / الأداء

| Metric / المقياس | Before P0 | After P0/P1/P2 |
|------------------|-----------|----------------|
| **Cold start (first search)** | 17,520 ms | **71 ms** ✅ |
| **Warm search (cached)** | — | **~0.1 ms** (cache hit) |
| **FTS5 Arabic** | — | **0.1 ms** |
| **FTS5 English** | — | **0.2 ms** |
| **Recall vs FTS5** | — | **100%** ✅ |
| **Edge cases** | — | **100% pass** ✅ |
| **Query cache hit rate** | — | **98%+** (repeated queries) |

### Key Optimizations / التحسينات الرئيسية

1. **P0: Background Preload** — BGE-M3 loads in a daemon thread during `initialize()`. The `threading.Event` ensures the first `prefetch()` waits for preload instead of blocking for 17 seconds.
2. **P1: Source Labels** — Each result shows clear provenance: `[graph:hermes-memory]` for facts, `[session:2h ago]` for session messages, `[graph:expanded·1hop]` for neighbor expansion.
3. **P2: Limit Accuracy** — Fixed `int()` cast on JSON input + `total` reported after slicing. `limit=5` now returns exactly 5 results with `total=5`.
4. **P3: Thread-Safe Cache** — `QueryResultCache` uses `threading.Lock()` with a `MAX_SIZE=500` cap. No race conditions, no memory leaks.
5. **RRF Re-ranking** — Reciprocal Rank Fusion merges graph and FTS5 results with configurable weights (graph ×1.5, fts5 ×1.0).

---

## 🔧 Configuration / الإعدادات

### Plugin Config (`~/.hermes/config.yaml`)

```yaml
memory:
  provider: unified

plugins:
  unified:
    graphs_dir: /home/anwar/.hermes/graphs
    embedding_model: BAAI/bge-m3
    device: cuda                  # cuda | cpu | auto
    similarity_threshold: 0.6     # Minimum cosine similarity
    chunk_size: 512               # Text chunk size (Arabic-aware)
    chunk_overlap: 96             # Overlap between chunks
    community_algorithm: louvain  # louvain | leiden
```

### Environment Variables / متغيرات البيئة

```bash
# Summarizer model (default: qwen2.5:3b)
export HERMES_SUMMARIZER_MODEL=qwen2.5:7b

# Ollama endpoint
export OLLAMA_URL=http://localhost:11434/api/chat
```

---

## 🛠️ API Reference / مرجع الواجهة

### Tools Available / الأدوات المتاحة

| Tool | Description | Required Args |
|------|-------------|---------------|
| `unified_search` | Hybrid search across all memory layers | `query` |
| `graph_search` | Semantic search in knowledge graph | `query`, `project` |
| `graph_index` | Index a directory into the knowledge graph | `path` |

### `unified_search`

```json
{
  "query": "كيف أضيف نموذج جديد؟",
  "limit": 5
}
```

**Response:**
```json
{
  "query": "كيف أضيف نموذج جديد؟",
  "results": [
    {
      "similarity": 0.87,
      "content": "المستخدم يريد إضافة نموذج qwen3.6-plus...",
      "type": "fact",
      "project": "hermes-memory",
      "source": "vector"
    }
  ],
  "total": 5
}
```

### `graph_search`

```json
{
  "query": "knowledge graph",
  "project": "hermes-memory",
  "top_k": 5
}
```

---

## 📊 Comparison / المقارنة

| Feature / الميزة | Original Hermes | Super Memory |
|------------------|-----------------|--------------|
| **Semantic search** | ❌ Not available | ✅ BGE-M3 cosine similarity |
| **Knowledge graph** | ❌ Not available | ✅ 6+ projects with embeddings |
| **100% local** | ⚠️ Cloud providers | ✅ Zero external calls |
| **Arabic normalization** | ⚠️ Basic FTS5 | ✅ Comprehensive (hamza, yaa, diacritics) |
| **Auto session summary** | ❌ Manual only | ✅ Every 60 minutes |
| **Fact extraction** | ❌ Not available | ✅ 9 classified categories |
| **RRF re-ranking** | ❌ Not available | ✅ graph×1.5 + fts5×1.0 |
| **Query caching** | ❌ Not available | ✅ TTL 5m, 98%+ hit rate |
| **Source labels** | ❌ Generic | ✅ `[graph:project]` / `[session:age]` |
| **Thread safety** | ⚠️ Basic | ✅ Event sync + Lock |
| **Cold start** | ~17s (first query) | **71ms** (background preload) |

---

## 🧪 Testing / الاختبارات

```bash
cd ~/.hermes/plugins
python3 -m pytest unified/tests/ -v
# Expected: 179 passed, 0 failed
```

### Coverage / التغطية

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_arabic_normalizer.py` | 43 | Arabic text normalization |
| `test_document_loader.py` | 34 | File loading with filters |
| `test_unified_memory.py` | 33 | Unified search pipeline |
| `test_graph_builder.py` | 29 | Graph construction |
| `test_graph_storage.py` | 25 | Compact JSON storage |
| `test_community_detector.py` | 15 | Community detection |

---

## 🔄 Upgrade from Previous Version / الترقية من إصدار سابق

After updating `graph_builder.py` (text normalization before embedding), old graphs need re-indexing:

```bash
# Rebuild the graph / إعادة بناء الرسم
python3 ~/.hermes/scripts/graph_updater.py

# Or from inside Hermes / أو من داخل هرمز
# graph_index(path="~/project", reindex=True)
```

> **Why?** Old nodes are stored without Arabic normalization in embeddings, so cosine similarity fails when searching with normalized text.  
> **لماذا؟** العقد القديمة مخزنة بدون تطبيع عربي في التضمينات، فـ cosine similarity يفشل عند البحث بالنص المطبّع.

---

## ❓ Troubleshooting / استكشاف الأخطاء

### BGE-M3 fails to load / فشل تحميل BGE-M3
```bash
# First download requires internet. After that, fully offline.
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-m3')"
```

### Memory returns no results / الذاكرة لا ترجع نتائج
```bash
# Rebuild the graph / إعادة بناء الرسم
rm -f ~/.hermes/graphs/hermes-memory/graph.json
echo '{"indexed_fact_hashes": []}' > ~/.hermes/memory/.graph_tracker.json
python3 ~/.hermes/scripts/graph_updater.py
```

### ImportError: text_splitter
```bash
# Verify file exists / تحقق من وجود الملف
ls ~/.hermes/plugins/unified/text_splitter.py
# If missing, reinstall / إذا غير موجود، أعد التثبيت
./install.sh
```

### Cold start still slow / التأخير الأول لا يزال بطيئاً
```bash
# Verify background preload is active / تحقق من التحميل الخلفي
tail -f ~/.hermes/logs/agent.log | grep "preload"
# Should show: "جاري تحميل نموذج التضمين: BAAI/bge-m3" during initialize()
```

---

## 📊 Current Metrics / المقاييس الحالية

| Metric | Value |
|--------|-------|
| **Tests** | 179 passed, 0 failures, ~6s |
| **hermes-memory graph** | 864 nodes, 8,058 edges |
| **Embeddings** | BGE-M3, 1024-dim, FP16 on CUDA |
| **Search recall** | 100% vs FTS5 baseline |
| **Query cache** | TTL 5m, MAX_SIZE 500 |
| **Indexed projects** | 12+ (hermes-memory, multica, wikillm-*) |
| **Fact categories** | 9 (technical, preference, project, config...) |

---

## 📜 License / الرخصة

**MIT License** — Use freely for any purpose.  
رخصة MIT — استخدمه بحرية لأي غرض.

---

## 🙏 Acknowledgments / الشكر

| Project | Role |
|---------|------|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Nous Research — Core agent framework |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | BAAI — Multilingual embedding model |
| [Qwen2.5](https://ollama.com/library/qwen2.5) | Alibaba — Local LLM for summarization |
| [Ollama](https://ollama.com) | Local LLM inference runtime |
| [NetworkX](https://networkx.org) | Knowledge graph operations |
| [NumPy](https://numpy.org) | Vectorized similarity computation |
| [SentenceTransformers](https://sbert.net) | BGE-M3 integration |

---

<p align="center">
  <em>Built with ❤️ for Hermes Agent — by Anwar Anabtawi</em><br>
  <em>مبني بـ ❤️ لـ Hermes Agent — بواسطة أنور عنبتاوي</em>
</p>
