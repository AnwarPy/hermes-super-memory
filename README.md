# 🧠 Hermes Super Memory

**ذاكرة خارقة لهرمز — 100% محلي، صفر اتصالات خارجية**

---

## ✨ المميزات

- 🤖 **تلخيص تلقائي** للجلسات (كل ساعة)
- 🔍 **استخراج حقائق** مصنفة (9 فئات)
- 🕸️ **رسم معرفي** للحقائق (BGE-M3 embeddings)
- 🔗 **بحث دلالي** يتكامل مع prefetch() تلقائياً
- 🇸🇦 **دعم عربي** ممتاز (trigram + normalization)
- 🔒 **خصوصية كاملة** — كل شيء على جهازك
- ⚡ **صفر تأثير** على سرعة الردود

## 📦 المتطلبات

- Hermes Agent مثبت ومُعد
- Python 3.10+
- Ollama (سيُثبّت تلقائياً)
- ~4GB RAM (BGE-M3 + qwen2.5:3b)
- ~3GB تخزين (نماذج + رسوم)

## 🚀 التثبيت

```bash
# 1. استنساخ المشروع
git clone https://github.com/YOUR_USERNAME/hermes-super-memory.git
cd hermes-super-memory

# 2. تشغيل المثبت
chmod +x install.sh
./install.sh

# 3. اختبار (اختياري)
./test.sh
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
│   ├── facts_auto/             # حقائق مصنفة
│   └── .summarizer_tracker.json
├── graphs/
│   └── hermes-memory/          # الرسم المعرفي
│       ├── graph.json
│       ├── communities.json
│       └── metadata.json
└── plugins/
    └── unified/                # plugin البحث الموحد
        ├── __init__.py
        ├── embedding_model.py
        └── tests/
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

## 🧪 الاختبارات

```bash
cd ~/.hermes/plugins
python3 -m pytest unified/tests/ -v
# يجب أن يُظهر: 136 passed
```

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

## 📜 الرخصة

MIT License — استخدمه كما تشاء.

## 🙏 شكراً

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Nous Research
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — BAAI
- [Qwen2.5](https://ollama.com/library/qwen2.5) — Alibaba
- [Ollama](https://ollama.com) — محلي LLM inference
