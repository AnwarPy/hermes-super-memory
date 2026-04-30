#!/bin/bash
# ============================================================
# Hermes Super Memory — Test Script
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo "═══════════════════════════════════════════════"
echo "   🧪 Hermes Super Memory Tests"
echo "═══════════════════════════════════════════════"
echo ""

# Test 1: Ollama
echo -e "${YELLOW}[1/5] Testing Ollama...${NC}"
if curl -s --connect-timeout 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Ollama running${NC}"
    
    # Check qwen2.5:3b
    if curl -s http://localhost:11434/api/tags | grep -q "qwen2.5"; then
        echo -e "${GREEN}  ✅ qwen2.5:3b available${NC}"
    else
        echo -e "${RED}  ❌ qwen2.5:3b not found. Run: ollama pull qwen2.5:3b${NC}"
        exit 1
    fi
else
    echo -e "${RED}  ❌ Ollama not running. Run: ollama serve${NC}"
    exit 1
fi

# Test 2: Scripts exist
echo -e "${YELLOW}[2/5] Checking scripts...${NC}"
for f in session_summarizer.py graph_updater.py fact_extractor.py; do
    if [ -f "$HERMES_HOME/scripts/$f" ]; then
        echo -e "${GREEN}  ✅ $f${NC}"
    else
        echo -e "${RED}  ❌ $f missing${NC}"
        exit 1
    fi
done

# Test 3: Plugin exists
echo -e "${YELLOW}[3/5] Checking plugin...${NC}"
if [ -f "$HERMES_HOME/plugins/unified/__init__.py" ]; then
    echo -e "${GREEN}  ✅ unified plugin${NC}"
else
    echo -e "${RED}  ❌ unified plugin missing${NC}"
    exit 1
fi

# Test 4: Unit tests
echo -e "${YELLOW}[4/5] Running unit tests...${NC}"
cd "$HERMES_HOME/plugins"
TEST_RESULT=$(python3 -m pytest unified/tests/ -v --tb=short -q 2>&1)
if echo "$TEST_RESULT" | grep -q "passed"; then
    PASSED=$(echo "$TEST_RESULT" | grep -oP '\d+ passed')
    echo -e "${GREEN}  ✅ $PASSED${NC}"
else
    echo -e "${RED}  ❌ Tests failed:${NC}"
    echo "$TEST_RESULT" | tail -10
    exit 1
fi

# Test 5: Quick pipeline test
echo -e "${YELLOW}[5/5] Quick pipeline test...${NC}"
python3 -c "
import sys, os
sys.path.insert(0, '$HERMES_HOME/plugins')
from unified.embedding_model import EmbeddingModel

# Test BGE-M3
model = EmbeddingModel(device='cpu', use_fp16=False)
emb = model.embed_query('test')
assert len(emb) == 1024, f'Expected 1024, got {len(emb)}'
print(f'  BGE-M3: {len(emb)}-dim ✅')
" 2>&1 | grep -v "^Loading\|^Warning\|^جاري\|^✓\|^الجهاز\|^  -"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ All tests passed!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
