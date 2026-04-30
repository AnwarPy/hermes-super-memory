# 🧠 Hermes Super Memory

**ذاكرة خارقة لـ Hermes Agent — 100% محلي، صفر اتصالات خارجية**

---

## ✨ المميزات

- 🤖 **تلخيص تلقائي** للجلسات مع كشف اللغة (عربي/إنجليزي)
- 🔍 **استخراج حقائق** مصنفة (9 فئات: technical, preference, project...)
- 🕸️ **رسم معرفي** للحقائق (BGE-M3 embeddings — 1024-dim)
- 🔗 **بحث دلالي vectorized** يتكامل مع `prefetch()` تلقائياً
- 🇸🇦 **دعم عربي متقدم** — تطبيع شامل (همزات، ياء، حروف أعجمية، حركات)
- 🔒 **خصوصية كاملة** — كل شيء على جهازك
- ⚡ **صفر تأثير** على سرعة الردود (cron في الخلفية)
- 🧵 **Thread-safe** — double-checked locking على التهيئة الكسولة

## 📦 المتطلبات

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) مثبت ومُعد
- Python 3.10+
- Ollama (سيُثبّت تلقائياً)
- ~4GB RAM (BGE-M3 + qwen2.5:3b)
- ~3GB تخزين (نماذج + رسوم)

## 🚀 التثبيت

```bash
# 1. استنساخ المشروع
git clone https://github.com/AnwarPy/hermes-super-memory.git
cd hermes-super-memory

# 2. تشغيل المثبت
chmod +x install.sh
./install.sh

# 3. اختبار (اختياري)
./test.sh

# 4. إعادة تشغيل Hermes
hermes gateway restart
```

## 📁 الهيكل بعد التثبيت

```
~/.hermes/
├── scripts/
│   ├── session_summarizer.py   # تلخيص الجلسات
│   ├── graph_updater.py        # تحديث الرسم المعرفي
│   └── fact_extractor.py       # استخراج الحقائق
├── memory/
│   ├── summaries/              # ملخصات الجلسات
│   ├── facts_auto/             # حقائق مصنفة (9 فئات JSONL)
│   └── .summarizer_tracker.json
├── graphs/
│   └── hermes-memory/          # الرسم المعرفي
│       ├── graph.json
│       ├── communities.json
│       └── metadata.json
└── plugins/
    └── unified/                # plugin البحث الموحد
        ├── __init__.py         # prefetch + RRF + vectorized search
        ├── embedding_model.py  # BGE-M3 integration
        ├── graph_builder.py    # بناء الرسوم مع تطبيع عربي
        ├── graph_storage.py    # تخزين الرسوم (compact JSON)
        ├── graph_engine.py     # محرك الرسوم
        ├── document_loader.py  # تحميل المستندات (فلاتر + حد حجم)
        ├── community_detector.py # اكتشاف المجتمعات (Leiden + Louvain)
        ├── arabic_normalizer.py # تطبيع عربي شامل
        ├── text_splitter.py    # تقسيم النصوص مع وعي بالعربية
        ├── synonym_dict.json   # قاموس المترادفات (معطيل افتراضياً)
        ├── plugin.yaml         # إعدادات
        └── tests/              # 179 اختبار
            ├── test_arabic_normalizer.py  # 43 اختبار عربي
            ├── test_graph_builder.py
            ├── test_document_loader.py
            ├── test_unified_memory.py
            ├── test_graph_storage.py
            └── test_community_detector.py
```

## ⚙️ التشغيل

### تلقائي (بعد التثبيت):
- **كل ساعة**: تلخيص الجلسات الجديدة
- **كل 6 ساعات**: استخراج حقائق + تحديث الرسم

### يدوي:
```bash
# تلخيص آخر الجلسات
python3 ~/.hermes/scripts/session_summarizer.py

# استخراج حقائق أعمق
python3 ~/.hermes/scripts/fact_extractor.py

# تحديث الرسم المعرفي
python3 ~/.hermes/scripts/graph_updater.py
```

## 🔧 التهيئة (Configuration)

### متغيرات البيئة:
```bash
# تغيير موديل التلخيص (افتراضي: qwen2.5:3b)
export HERMES_SUMMARIZER_MODEL=qwen2.5:7b

# تغيير عنوان Ollama
export OLLAMA_URL=http://localhost:11434/api/chat

# تفعيل توسيع المترادفات (معطيل افتراضياً)
# في plugin.yaml:
# enable_synonym_expansion: true
```

### الموديلات المدعومة:
| الموديل | الحجم | السرعة | الجودة | ملاحظات |
|---------|-------|--------|--------|---------|
| qwen2.5:3b | 1.9GB | ⚡ 1.8s | 🟡 مقبول | **الافتراضي** — أفضل توازن |
| qwen2.5:7b | 4.7GB | 🟡 5.3s | 🟡 مقبول | أبطأ، إنجليزي أحياناً |
| qwen2.5:14b | 9GB | 🔴 10.7s | ✅ جيد | أدق بس أبطأ |

## 🧪 الاختبارات

```bash
cd ~/.hermes/plugins
python3 -m pytest unified/tests/ -v
# يجب أن يُظهر: 179 passed
```

### تغطية الاختبارات:
- **test_arabic_normalizer.py** — 43 اختبار (تطبيع عربي شامل)
- **test_graph_builder.py** — 28 اختبار (بناء الرسوم)
- **test_document_loader.py** — 34 اختبار (تحميل المستندات)
- **test_unified_memory.py** — 43 اختبار (البحث الموحد)
- **test_graph_storage.py** — 25 اختبار (التخزين)
- **test_community_detector.py** — 15 اختبار (اكتشاف المجتمعات)

## 🏗️ المعمارية

```
┌─────────────────────────────────────────────────┐
│              prefetch() — Unified Memory          │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   FTS5   │  │  Graph   │  │  Memory  │       │
│  │ (رسائل)  │  │(wikillm) │  │(حقائق)   │       │
│  └────┬─────┘  └────┬─────┘  └──────────┘       │
│       │              │                             │
│       │    ┌─────────┴──────────┐                │
│       │    │  hermes-memory     │ ← الجديد       │
│       │    │  (حقائق مستخرجة)  │                 │
│       │    └───────────────────┘                 │
│       │              │                             │
│       ▼              ▼                             │
│  ┌─────────────────────────┐                     │
│  │   RRF Re-ranking        │                     │
│  │   (graph×1.5 + fts5×1)  │                     │
│  └─────────────────────────┘                     │
└─────────────────────────────────────────────────┘

[كل ساعة] Session Summarizer (Ollama محلي)
    ↓ يلخص الجلسات → summaries/*.json
    ↓ يستخرج حقائق → facts_auto/*.jsonl

[كل 6 ساعات] Fact Extractor + Graph Updater (BGE-M3)
    ↓ يستخرج حقائق أعمق
    ↓ يبني hermes-memory بـ BGE-M3 embeddings
    ↓ يُصبح قابل للبحث تلقائياً
```

## 🔍 مقارنة مع النظام الأصلي

| الميزة | Hermes الأصلي | Super Memory |
|--------|-------------|-------------|
| **البحث الدلالي** | ❌ لا يوجد | ✅ BGE-M3 cosine |
| **الرسم المعرفي** | ❌ لا يوجد | ✅ 6 مشاريع + hermes-memory |
| **graph_search tool** | ❌ غير موجود | ✅ vectorized |
| **التضمين المتجهي** | ❌ (HRR رياضي) | ✅ BGE-M3 1024-dim |
| **تلخيص الجلسات** | ❌ يدوي فقط | ✅ تلقائي + كشف لغة |
| **استخراج الحقائق** | ❌ لا يوجد | ✅ 9 فئات |
| **التطبيع العربي** | ⚡ FTS5 أساسي | ✅ شامل (همزات/ياء/أعجمي) |
| **الأداء** | بحث FTS5 فقط | ✅ vectorized + cache 0ms |
| **المزودات** | 8 سحابية | ✅ 0 (كله محلي) |
| **الخصوصية** | ⚠️ بيانات للسحابة | ✅ كل شيء على جهازك |
| **التكلفة** | ⚠️ API keys | ✅ صفر |

## ❓ استكشاف الأخطاء

### Ollama لا يعمل
```bash
ollama serve  # شغّل Ollama
ollama pull qwen2.5:3b  # نزّل الموديل
```

### BGE-M3 لا يتحمل
```bash
# تأكد من الإنترنت أول مرة (ينزل ~2.3GB)
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### الذاكرة لا تظهر نتائج
```bash
# أعد بناء الرسم
rm ~/.hermes/graphs/hermes-memory/graph.json
echo '{"indexed_fact_hashes": []}' > ~/.hermes/memory/.graph_tracker.json
python3 ~/.hermes/scripts/graph_updater.py
```

### ImportError: text_splitter
```bash
# تأكد من وجود الملف
ls ~/.hermes/plugins/unified/text_splitter.py
# إذا غير موجود، أعد التثبيت
./install.sh
```

## 📊 المقاييس

- **179 اختبار** — 6.1s, 0 failures
- **168 عقدة** في hermes-memory (1024-dim BGE-M3)
- **608 حواف** (similar + session + category)
- **9 فئات** حقائق
- **Vectorized search** — 30-80x أسرع من O(N) loop
- **BGE-M3** — يتحمل في ~13s (CPU), ~382MB RAM

## 📜 الرخصة

MIT License — استخدمه كما تشاء.

## 🙏 شكراً

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Nous Research
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — BAAI
- [Qwen2.5](https://ollama.com/library/qwen2.5) — Alibaba
- [Ollama](https://ollama.com) — محلي LLM inference
- [NetworkX](https://networkx.org) — رسوم معرفية
- [NumPy](https://numpy.org) — vectorized similarity
