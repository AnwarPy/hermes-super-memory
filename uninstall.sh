#!/bin/bash
# ============================================================
# Hermes Super Memory — Uninstaller
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo -e "${RED}═══════════════════════════════════════════════${NC}"
echo -e "${RED}   🗑️  Hermes Super Memory Uninstaller${NC}"
echo -e "${RED}═══════════════════════════════════════════════${NC}"
echo ""

read -p "هل أنت متأكد؟ (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "تم الإلغاء."
    exit 0
fi

# Remove cron jobs
echo -e "${YELLOW}Removing cron jobs...${NC}"
crontab -l 2>/dev/null | grep -v "run_summarizer\|run_graph_updater" | crontab -
echo -e "${GREEN}  ✅ Cron jobs removed${NC}"

# Remove scripts
echo -e "${YELLOW}Removing scripts...${NC}"
rm -f "$HERMES_HOME/scripts/session_summarizer.py"
rm -f "$HERMES_HOME/scripts/graph_updater.py"
rm -f "$HERMES_HOME/scripts/fact_extractor.py"
rm -f "$HERMES_HOME/scripts/run_summarizer.sh"
rm -f "$HERMES_HOME/scripts/run_graph_updater.sh"
echo -e "${GREEN}  ✅ Scripts removed${NC}"

# Remove plugin
echo -e "${YELLOW}Removing unified plugin...${NC}"
rm -rf "$HERMES_HOME/plugins/unified"
echo -e "${GREEN}  ✅ Plugin removed${NC}"

# Remove memory data (ask first)
read -p "حذف البيانات (ملخصات + حقائق + رسوم)؟ (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$HERMES_HOME/memory/summaries"
    rm -rf "$HERMES_HOME/memory/facts_auto"
    rm -f "$HERMES_HOME/memory/.summarizer_tracker.json"
    rm -f "$HERMES_HOME/memory/.graph_tracker.json"
    rm -f "$HERMES_HOME/memory/.extractor_tracker.json"
    rm -rf "$HERMES_HOME/graphs/hermes-memory"
    echo -e "${GREEN}  ✅ Data removed${NC}"
else
    echo -e "${YELLOW}  ⏭️  Data preserved${NC}"
fi

# Remove Ollama models (ask first)
read -p "حذف نماذج Ollama (qwen2.5:3b, nomic-embed-text)؟ (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ollama rm qwen2.5:3b 2>/dev/null || true
    ollama rm nomic-embed-text:v1.5 2>/dev/null || true
    echo -e "${GREEN}  ✅ Models removed${NC}"
else
    echo -e "${YELLOW}  ⏭️  Models preserved${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Hermes Super Memory uninstalled${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}💡 أعد تشغيل Hermes: hermes gateway restart${NC}"
