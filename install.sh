#!/bin/bash
# ============================================================
# Hermes Super Memory — Installer
# 100% local memory pipeline for Hermes Agent
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🧠 Hermes Super Memory Installer${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo ""

# ============================================================
# Step 1: Check prerequisites
# ============================================================
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check Hermes
if [ ! -d "$HERMES_HOME" ]; then
    echo -e "${RED}❌ Hermes not found at $HERMES_HOME${NC}"
    echo "   Install Hermes first: https://github.com/NousResearch/hermes-agent"
    exit 1
fi
echo -e "${GREEN}  ✅ Hermes found${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ Python3: $(python3 --version)${NC}"

# ============================================================
# Step 2: Install Ollama + Models
# ============================================================
echo -e "${YELLOW}[2/6] Setting up Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo -e "${GREEN}  ✅ Ollama installed${NC}"

# Start Ollama if not running
if ! curl -s --connect-timeout 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  Starting Ollama..."
    ollama serve &> /dev/null &
    sleep 3
fi

# Pull models
echo "  Pulling qwen2.5:3b (summarization)..."
ollama pull qwen2.5:3b

echo "  Pulling nomic-embed-text (embeddings backup)..."
ollama pull nomic-embed-text:v1.5 || true

echo -e "${GREEN}  ✅ Models ready${NC}"

# ============================================================
# Step 3: Install Python dependencies
# ============================================================
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"

pip_install() {
    python3 -c "import $1" 2>/dev/null || python3 -m pip install "$2" -q
}

pip_install sentence_transformers sentence-transformers
pip_install networkx networkx
pip_install numpy numpy

echo -e "${GREEN}  ✅ Dependencies installed${NC}"

# ============================================================
# Step 4: Copy scripts
# ============================================================
echo -e "${YELLOW}[4/6] Installing scripts...${NC}"

mkdir -p "$HERMES_HOME/scripts"
mkdir -p "$HERMES_HOME/memory/summaries"
mkdir -p "$HERMES_HOME/memory/facts_auto"
mkdir -p "$HERMES_HOME/graphs/hermes-memory"

cp "$SCRIPT_DIR/scripts/session_summarizer.py" "$HERMES_HOME/scripts/"
cp "$SCRIPT_DIR/scripts/graph_updater.py" "$HERMES_HOME/scripts/"
cp "$SCRIPT_DIR/scripts/fact_extractor.py" "$HERMES_HOME/scripts/"

echo -e "${GREEN}  ✅ Scripts installed${NC}"

# ============================================================
# Step 5: Install unified plugin
# ============================================================
echo -e "${YELLOW}[5/6] Installing unified memory plugin...${NC}"

PLUGIN_DIR="$HERMES_HOME/plugins/unified"
mkdir -p "$PLUGIN_DIR/tests"

# Copy plugin files
cp "$SCRIPT_DIR/plugins/unified/__init__.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/embedding_model.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/graph_builder.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/graph_storage.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/graph_engine.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/document_loader.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/community_detector.py" "$PLUGIN_DIR/"
cp "$SCRIPT_DIR/plugins/unified/arabic_normalizer.py" "$PLUGIN_DIR/"

# Copy tests
if [ -d "$SCRIPT_DIR/plugins/unified/tests" ]; then
    cp "$SCRIPT_DIR/plugins/unified/tests/"*.py "$PLUGIN_DIR/tests/"
fi

# Copy config
if [ -f "$SCRIPT_DIR/plugins/unified/plugin.yaml" ]; then
    cp "$SCRIPT_DIR/plugins/unified/plugin.yaml" "$PLUGIN_DIR/"
fi

# Copy synonym dict if exists
if [ -f "$SCRIPT_DIR/plugins/unified/synonym_dict.json" ]; then
    cp "$SCRIPT_DIR/plugins/unified/synonym_dict.json" "$PLUGIN_DIR/"
fi

echo -e "${GREEN}  ✅ Plugin installed${NC}"

# ============================================================
# Step 6: Setup cron jobs
# ============================================================
echo -e "${YELLOW}[6/6] Setting up cron jobs...${NC}"

# Create cron wrapper scripts
cat > "$HERMES_HOME/scripts/run_summarizer.sh" << 'EOF'
#!/bin/bash
cd ~/.hermes
python3 scripts/session_summarizer.py >> ~/.hermes/logs/summarizer.log 2>&1
EOF

cat > "$HERMES_HOME/scripts/run_graph_updater.sh" << 'EOF'
#!/bin/bash
cd ~/.hermes
python3 scripts/fact_extractor.py >> ~/.hermes/logs/graph_updater.log 2>&1
python3 scripts/graph_updater.py >> ~/.hermes/logs/graph_updater.log 2>&1
EOF

chmod +x "$HERMES_HOME/scripts/run_summarizer.sh"
chmod +x "$HERMES_HOME/scripts/run_graph_updater.sh"

# Add to crontab (if not already there)
CRON_SUMMARIZER="0 * * * * $HERMES_HOME/scripts/run_summarizer.sh"
CRON_UPDATER="0 */6 * * * $HERMES_HOME/scripts/run_graph_updater.sh"

(crontab -l 2>/dev/null | grep -v "run_summarizer\|run_graph_updater"; echo "$CRON_SUMMARIZER"; echo "$CRON_UPDATER") | crontab -

echo -e "${GREEN}  ✅ Cron jobs configured${NC}"

# ============================================================
# Done
# ============================================================
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Hermes Super Memory installed!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "📋 ما تم تثبيته:"
echo -e "   • 3 سكريبتات في ${BLUE}~/.hermes/scripts/${NC}"
echo -e "   • unified plugin في ${BLUE}~/.hermes/plugins/unified/${NC}"
echo -e "   • Ollama + qwen2.5:3b (1.9GB)"
echo -e "   • BGE-M3 (ينزل تلقائياً عند أول تشغيل)"
echo ""
echo -e "📋 Cron Jobs:"
echo -e "   • كل ساعة: تلخيص الجلسات"
echo -e "   • كل 6 ساعات: استخراج حقائق + تحديث الرسم"
echo ""
echo -e "📋 الأوامر:"
echo -e "   • اختبار: ${BLUE}cd ~/.hermes/plugins && python3 -m pytest unified/tests/ -v${NC}"
echo -e "   • تشغيل يدوي: ${BLUE}python3 ~/.hermes/scripts/session_summarizer.py${NC}"
echo -e "   • السجلات: ${BLUE}tail -f ~/.hermes/logs/summarizer.log${NC}"
echo ""
echo -e "${YELLOW}💡 أول تشغيل: BGE-M3 ينزل ~2.3GB (مرة واحدة فقط)${NC}"
echo -e "${YELLOW}💡 أعد تشغيل Hermes: hermes gateway restart${NC}"
